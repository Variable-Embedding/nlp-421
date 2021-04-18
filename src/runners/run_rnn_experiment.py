
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

def run_rnn_experiment(epochs=2, enable_mp=True, device="gpu", **nn_data):
    """A function wrapper to execute training.
    :param epochs: integer, number of iterations to run.
    :param enable_mp: bool, default to True. whether to enable multiprocessing
    :param device: string, default to "gpu" but only runs on gpu if detected, else "cpu".
    :param nn_data: a dict of data containing required data elements for nn training.
    :return: True if function completes.
    """
    results = Results()

    stages = nn_data.keys()
    logging.info(f'Preparing experiment for {stages}.')

    model = Model(**nn_data['train'], device=device)
    model.to(model.device)

    train_data = nn_data['train']
    # TODO: valid and test integration
    valid_data = nn_data['valid']
    test_data = nn_data['test']

    num_iters = report_model_parameters(model=model, tokens=train_data['tokens'])
    start_time = time.time()

    logging.info(f'===== Starting Training for {epochs}x Epochs, {model.batch_size}x batches'
                 f', {model.embedding_size} embedding size'
                 f', and {model.dictionary_size} dictionary size'
                 f' with Device: {model.device}. =====')

    total_epochs = tqdm(range(epochs), desc="Training Progress", leave=True, position=0, total=epochs)

    for epoch in total_epochs:
        train_epoch(model=model, curr_epoch=epoch, total_epochs=epochs, tokens=train_data["tokens"], num_iters=num_iters, enable_mp=enable_mp, results=results)

    end_time = time.time()
    elapsed_time = end_time-start_time
    display_hrs = elapsed_time / 3600
    logging.info(f'===== Finished Training. '
                 f'Elapsed time with enable_mp={enable_mp} is {round(display_hrs, 4)} hours. =====')
    # FIXME: Configure results class to capture training statistics.
    # logging.info(f'Results {len(results.train_records)}')

    return True


def train_epoch(model, curr_epoch, total_epochs, num_iters, learning_rate=1, learning_rate_decay=1, tokens=None, train_dataloader=None, display_frequency=0.02, enable_mp=True, results=None):
    """Apply training to a single epoch.

    :param model: The PyTorch model class.
    :param curr_epoch: integer, the current epoch.
    :param total_epochs: integer, max number of epochs to train.
    :param num_iters: integer, estimated number of iterations based on tokens and batch size.
    :param learning_rate: int or float, Default to 1. the learning rate to clip gradients
    :param learning_rate_decay: int or float, Default to 1. rate at which to decay learning rate.
    :param tokens: a torch data struct of numbers representing a corpus of strings.
    :param train_dataloader: a torch object that returns data set objects. Optional.
    :param display_frequency: float, the rate at which to display tqdm progress bar. Default to 2% of total expected iterations.
    :param enable_mp: bool, Default to True to enable multiprocessing during each epoch.
    :param results: a Results() class to store training results.
    :return: None.
    """
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
            p = mp.Process(target=_train_epoch, args=(model
                                                      , tokens
                                                      , epoch_counter
                                                      , display_interval
                                                      , learning_rate
                                                      , learning_rate_decay
                                                      , curr_epoch
                                                      , total_epochs
                                                      , num_iters
                                                      , rank
                                                      , num_procs
                                                      , results)
                           )
            p.start()
            processes.append(p)
        for p in tqdm(processes, desc='Joining MP processes.'):
            p.join()

    else:
        _train_epoch(model
                     , tokens=tokens
                     , epoch_counter=epoch_counter
                     , display_interval=display_interval
                     , learning_rate=learning_rate
                     , learning_rate_decay=learning_rate_decay
                     , curr_epoch=curr_epoch
                     , total_epochs=total_epochs
                     , num_iters=num_iters
                     , results=results
                     )


def _train_epoch(model, tokens, epoch_counter, display_interval, learning_rate, learning_rate_decay, curr_epoch, total_epochs, num_iters, rank=None, num_procs=None, results=None):
    """Inner function to execute training loops.

    :params: Inherit params from parent function train_epoch()

    :return: None. Execute train loops, report statistics, and update results class.
    """

    pbar_desc = f'EPOCH: {epoch_counter} - PROC: {rank}' if rank is not None else f'EPOCH: {epoch_counter}'
    total_iters = num_iters // 2 if num_procs is not None else num_iters
    pbar_leave = False
    pbar_pos = None if rank else 0
    epoch_progress = tqdm(batch_data(tokens=tokens, model=model)
                          , desc=pbar_desc, leave=pbar_leave, total=total_iters, position=pbar_pos)
    epoch_loss = []
    rank = 0 if rank is None else rank
    curr_perplexity = 0

    for idx, (x, y) in enumerate(epoch_progress):

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

    :param tokens: the data to batch.
    :param model: the model to batch for.
    :param batch_size: the batch size, if None will use model.batch_size.
    :param sequence_step_size: the sequence step size.
    :param sequence_length: length of token sequence
    :param shuffle: Whether to shuffle the order of sequences.

    :return: Iterator for batched data.

    primary_reference: https://github.com/iryzhkov/nlp-pipeline
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


def report_model_parameters(model, tokens):
    """Report general model statistics such as paramters and number of iterations.

    :param model: The PyTorch model class.
    :param tokens: a 1D list or array of tokens in the target corpus.
    :return: integer, number of estimated iterations.

    primary_reference: https://github.com/iryzhkov/nlp-pipeline
    """

    num_iters = len(tokens) // model.batch_size // model.sequence_step_size
    num_iters += len(tokens) // model.batch_size // model.sequence_length

    num_parameters = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info("Number of model parameters: {} and Total Iterations: {}".format(num_parameters, num_iters))

    return num_iters


def loss_function(output, target):
    """Loss function for the model.

    :param output: the output of the model.
    :param target: the expected tokens.

    :return: Loss.

    primary_reference: https://github.com/iryzhkov/nlp-pipeline
    """
    return F.cross_entropy(output.reshape(-1, output.size(2)), target.reshape(-1)) * target.size(1)


class Results:
    def __init__(self):
        self.train_records = []

    def append_train_result(self, stage, pid, process, iteration, loss, perplexity, epoch):
        res = tuple((stage, pid, process, iteration, loss, perplexity, epoch))
        self.train_records.append(res)
