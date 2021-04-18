
from src.models.rnn_model import Model
from src.stages.stage_train_rnn_model import stage_train_rnn_model
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.multiprocessing as mp
from collections import defaultdict
import os
import time

def run_rnn_experiment(epochs=2, enable_mp=True, **nn_data):
    """
    """
    results = Results()

    stages = nn_data.keys()
    logging.info(f'Preparing experiment for {stages}.')

    model = Model(**nn_data['train'])
    model.to(model.device)

    train_data = nn_data['train']
    # TODO: valid and test integration
    valid_data = nn_data['valid']
    test_data = nn_data['test']

    num_iters = report_model_parameters(model, train_data['tokens'])
    start_time = time.time()

    logging.info(f'Starting Training for {epochs}x Epochs, {model.batch_size}x batches'
                 f', {model.embedding_size} embedding size'
                 f', and {model.dictionary_size} dictionary size'
                 f' with Device: {model.device}.')

    total_epochs = tqdm(range(epochs), desc="Training Progress", leave=True, position=0)

    for epoch in total_epochs:
        train_epoch(model=model, curr_epoch=epoch, total_epochs=epochs, tokens=train_data["tokens"], num_iters=num_iters, enable_mp=enable_mp, results=results)
    end_time = time.time()
    elapsed_time = end_time-start_time
    display_hrs = elapsed_time / 3600
    logging.info(f'Finished Training. Elapsed time with enable_mp={enable_mp} is {display_hrs} hours.')
    logging.info(f'Results {len(results.train_records)}')

    return True


def train_epoch(model, curr_epoch, total_epochs, num_iters, learning_rate=1, learning_rate_decay=1, tokens=None, train_dataloader=None, display_frequency=0.02, enable_mp=True, results=None):
    model.train()
    epoch_counter = curr_epoch+1
    epoch_loss = []

    if num_iters < 2000:
        display_interval = 50
    else:
        display_interval = int(num_iters * display_frequency)

    logging.info(f'Updating Statistics every {display_interval} iterations.')

    mp.set_start_method('spawn', force=True)

    if enable_mp:
        model.share_memory()
        # nun_procs -> how many parallel threads to run
        num_procs = 2  # alternatively use mp.cpu_count() to use all available threads
        processes = []
        # assign processes
        for rank in range(num_procs):
            p = mp.Process(target=_train_epoch, args=(model, tokens, epoch_counter, display_interval, learning_rate, learning_rate_decay, curr_epoch, total_epochs, num_iters, rank, num_procs,results))
            p.start()
            processes.append(p)
        for p in tqdm(processes, desc='Joining MP processes.'):
            p.join()

    else:
        _train_epoch(model, tokens, epoch_counter, display_interval, learning_rate, learning_rate_decay, curr_epoch, total_epochs, num_iters, results)


def _train_epoch(model, tokens, epoch_counter, display_interval, learning_rate, learning_rate_decay, curr_epoch, total_epochs, num_iters, rank=None, num_procs=None, results=None):

    pbar_desc = f'EPOCH: {epoch_counter} - PROC: {rank}' if rank is not None else f'EPOCH: {epoch_counter}'
    total_iters = (num_iters * total_epochs) // 2 if num_procs is not None else num_iters * total_epochs
    epoch_progress = tqdm(batch_data(tokens=tokens, model=model)
                          , desc=pbar_desc, leave=True, total=total_iters)
    epoch_loss = []

    curr_perplexity = 0
    counter = 0
    for idx, (x, y) in enumerate(epoch_progress):
        counter += 1

        pid = os.getpid()

        model.init_hidden()
        model.zero_grad()
        output = model(x)
        output = output.to(model.device)
        loss = loss_function(output, y)
        batch_loss = loss.item() / model.batch_size

        if idx == 0:
            curr_perplexity = np.exp(batch_loss)
            epoch_progress.set_description(
                'EPOCH: {} - PROC: {} - Start Perplexity: {:.2f} - Start Loss: {:.2f}'.format(epoch_counter, rank, curr_perplexity, batch_loss))
            epoch_progress.refresh()

        if idx > 0 and idx % display_interval == 0:
            curr_perplexity = np.exp(batch_loss)
            epoch_progress.set_description(
                'EPOCH: {} - PROC: {} - Curr  Perplexity: {:.2f} - Loss: {:.2f}'.format(epoch_counter, rank, curr_perplexity, batch_loss))
            epoch_progress.refresh()

        loss.backward()

        with torch.no_grad():
            norm = nn.utils.clip_grad_norm_(model.parameters(), model.max_norm)
            for param in model.parameters():
                lr = learning_rate * (learning_rate_decay ** curr_epoch)
                param -= lr * param.grad

        epoch_loss.append(batch_loss)

        results.append_train_result(stage='train', pid=pid, process=rank, iteration=idx, loss=batch_loss, perplexity=curr_perplexity, epoch=curr_epoch)


def batch_data(tokens, model, batch_size=None, sequence_length=None, sequence_step_size=None, shuffle=False):
    """Helper function to batch the data.

    Args:
        tokens: the data to batch.
        model: the model to batch for.
        batch_size: the batch size, if None will use model.batch_size.
        sequence_step_size: the sequence step size.
        sequence_length: length of token sequence
        shuffle: Whether to shuffle the order of sequences.

    Returns:
        Iterator for batched data.
    """
    if batch_size is None:
        batch_size = model.batch_size
    if sequence_length is None:
        sequence_length = model.sequence_length
    if sequence_step_size is None:
        sequence_step_size = model.sequence_step_size

    data = torch.tensor(tokens, dtype=torch.int64)
    words_per_batch = data.size(0) // batch_size
    data = data[:words_per_batch * batch_size]
    data = data.view(batch_size, -1)

    sequence_start_list = list(range(0, data.size(1) - sequence_length - 1, sequence_step_size))
    if shuffle:
        np.random.shuffle(sequence_start_list)

    for sequence_start in sequence_start_list:
        sequence_end = sequence_start + sequence_length
        prefix = data[:,sequence_start:sequence_end].transpose(1, 0).to(model.device)
        target = data[:,sequence_start + 1:sequence_end + 1].transpose(1, 0).to(model.device)
        yield prefix, target
        del prefix
        del target


class LanguageModelSequence(Dataset):
    # FIXME: Need to account for batch size when returning sequences, use batch_data() func for now.
    def __init__(self, model, data, sequence_length=None, sequence_step_size=None):

        self.data = data

        self.sequence_length = model.sequence_length if sequence_length is None else sequence_length
        self.sequence_step_size = model.sequence_step_size if sequence_step_size is None else sequence_step_size

        self.model=model

    def __getitem__(self, index):
        x = self.data[index: index + self.sequence_length]

        y = self.data[index + self.sequence_length: index + self.sequence_length + self.sequence_step_size]

        return torch.tensor(x, dtype=torch.long).to(self.model.device), torch.tensor(y, dtype=torch.long).to(self.model.device)

    def __len__(self):
        return len(self.data)


def report_model_parameters(model, train_data):

    num_iters = len(train_data) // model.batch_size // model.sequence_step_size
    num_iters += len(train_data) // model.batch_size // model.sequence_length

    num_parameters = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info("Number of model parameters: {} and Total Iterations: {}".format(num_parameters, num_iters))

    return num_iters


def loss_function(output, target):
    """Loss function for the model.

    Args:
        output: the output of the model.
        target: the expected tokens.

    Returns:
        Loss.
    """
    return F.cross_entropy(output.reshape(-1, output.size(2)), target.reshape(-1)) * target.size(1)


class Results:
    def __init__(self):
        self.train_records = []

    def append_train_result(self, stage, pid, process, iteration, loss, perplexity, epoch):
        res = tuple((stage, pid, process, iteration, loss, perplexity, epoch))
        self.train_records.append(res)
