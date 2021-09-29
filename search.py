from util.load_cfg import load_cfg

import click

from optimizer.EA.ea_agent import EvoAgent
from optimizer.EA.util.callback import IGDMonitor, HyperVolumeMonitor, NonDominatedProgress, TimeLogger, CheckpointSaver

from tensorboardX import SummaryWriter

@click.command()
@click.option('--config', '-cfg', required=True)
@click.option('--seed', '-s', default=0, type=int)
@click.option('--summary_writer', '-sw', default=True, type=bool)
@click.option('--console_log', default=True, type=bool)
def cli(config, seed, summary_writer, console_log):
    cfg = load_cfg(config, seed, console_log)

    if summary_writer:
        summary_writer = SummaryWriter(cfg.summary_dir)
    else:
        summary_writer = None

    callbacks = [
        NonDominatedProgress(plot_pf=False, labels=['Floating-point operations (M)', 'Error rate (%)']),
        IGDMonitor(normalize=True, from_archive=True, convert_to_pf_space=True, topk=5),
        HyperVolumeMonitor(normalize=True, topk=5, from_archive=True, convert_to_pf_space=True, ref_point=[1.1, 1.1]),
        CheckpointSaver(),
        TimeLogger()
    ]
    agent = EvoAgent(
        cfg, 
        seed,
        callbacks=callbacks,
        summary_writer=summary_writer
    )

    agent.solve()

if __name__ == '__main__':
    cli(['-cfg', 'config/baseline_moenas-201.yml'])

    