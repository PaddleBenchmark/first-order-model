from tqdm import trange
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater

from timer import TimeAverager
import time

def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    batch_cost_averager = TimeAverager()
    reader_cost_averager = TimeAverager()

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            iters = iter(dataloader)
            while 1:
                try:
                    start_time = step_start_time = time.time()
                    x = next(iters)
                    reader_cost_averager.record(time.time() - step_start_time)
                    losses_generator, generated = generator_full(x)

                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_generator.step()
                    optimizer_generator.zero_grad()
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()

                    if train_params['loss_weights']['generator_gan'] != 0:
                        optimizer_discriminator.zero_grad()
                        losses_discriminator = discriminator_full(x, generated)
                        loss_values = [val.mean() for val in losses_discriminator.values()]
                        loss = sum(loss_values)

                        loss.backward()
                        optimizer_discriminator.step()
                        optimizer_discriminator.zero_grad()
                    else:
                        losses_discriminator = {}

                    losses_generator.update(losses_discriminator)
                    losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                    logger.log_iter(losses=losses)

                    batch_cost_averager.record(time.time() - step_start_time,
                                       num_samples=train_params['batch_size'])

                    message = ""
                    data_time = reader_cost_averager.get_average()
                    step_time = batch_cost_averager.get_average()
                    ips = batch_cost_averager.get_ips_average()
                    message += 'batch_cost: %.5f sec ' % step_time
                    message += 'reader_cost: %.5f sec ' % data_time
                    message += 'ips: %.5f images/s ' % ips
                    logger.log_file.flush()
                    batch_cost_averager.reset()
                    print(message)
                except StopIteration:
                    break

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)
