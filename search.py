from util.load_cfg import load_cfg

import click

from optimizer.EA.ea_agent import EvoAgent
from optimizer.EA.util.callback import IGDMonitor, HyperVolumeMonitor, NonDominatedProgress, TimeLogger, CheckpointSaver

from tensorboardX import SummaryWriter

@click.command()
@click.option('--config', '-cfg', required=True)
@click.option('--seed', '-s', default=-1, type=int)
@click.option('--summary_writer', '-sw', default=True, type=bool)
@click.option('--console_log', default=True, type=bool)
@click.option("--loops_if_rand", type=int, default=30, help="Total runs for evaluation.")
@click.option('--population size', '-p', default=50, type=int)
@click.option('--n_evals', default=3000, type=int)
def cli(config, seed, summary_writer, console_log):
    cfg = load_cfg(config, seed, console_log)
    summary_writer = SummaryWriter(cfg.summary_dir) if summary_writer else None

    callbacks = [
        NonDominatedProgress(plot_pf=False, labels=['Floating-point operations (M)', 'Error rate (%)']),
        IGDMonitor(
            normalize=True, 
            from_archive=True, 
            convert_to_pf_space=True, 
            topk=5
        ),
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
    cli()

    