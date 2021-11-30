from util.load_cfg import load_cfg

import click

from optimizer.EA.ea_agent import EvoAgent
from optimizer.EA.util.callback import IGDMonitor, NonDominatedProgress, TimeLogger, CheckpointSaver

from tensorboardX import SummaryWriter

from copy import deepcopy

@click.command()
@click.option('--config', '-cfg', required=True, help='Provide the config files.')
@click.option('--summary_writer', '-sw', is_flag=True, help='Use summary writer to log graphs.')
@click.option('--console_log', is_flag=True, help='Log output to the console.')
@click.option("--loops_if_rand", type=int, default=10, help="Total runs for evaluation.")
@click.option('--seed', '-s', default=-1, type=int, help='Random seed.')
@click.option('--pop_size', '-p', default=-1, type=int, help='Population size')
@click.option('--n_evals', default=-1, type=int, help='Number of evaluations.')
@click.option('--use_archive', is_flag=True, help='Use elitist archive to evaluate for IGD instead of rank 0 in the population.')
@click.option('--eval_igd', is_flag=True, help='Calculate IGD each generation during the search.')
def cli(config, 
        console_log,
        loops_if_rand,
        seed,
        **kwargs):
    if seed < 0:
        try:
            for i in range(loops_if_rand):
                cfg = load_cfg(config, seed=i, console_log=console_log)
                solver = setup_agent(config=cfg, seed=i, **kwargs)
                solver.run()
        except KeyboardInterrupt:
            print('Interrupted. You have entered CTRL+C...')
        except Exception as e:
            import traceback
            traceback.print_exc()
    else:
        cfg = load_cfg(config, seed=seed, console_log=console_log)
        solver = setup_agent(config=cfg, seed=seed, **kwargs)
        solver.solve()

def setup_agent(config, seed, summary_writer, pop_size, n_evals, use_archive, eval_igd):
    cfg = deepcopy(config)
    if pop_size > 0:
        cfg.algorithm.kwargs.pop_size = pop_size
        cfg.algorithm.kwargs.n_offsprings = pop_size
    if n_evals > 0:
        cfg.termination.kwargs.n_max_evals = n_evals

    summary_writer = SummaryWriter(cfg.summary_dir) if summary_writer else None

    callbacks = [
        NonDominatedProgress(plot_pf=False, labels=['Floating-point operations (M)', 'Error rate (%)']),
        CheckpointSaver(),
        TimeLogger()
    ]

    if eval_igd:
        igd_monitor = IGDMonitor(
            normalize=True, 
            from_archive=use_archive, 
            convert_to_pf_space=True, 
            topk=5
        )
        callbacks = [igd_monitor] + callbacks

    agent = EvoAgent(
        cfg, 
        seed,
        callbacks=callbacks,
        summary_writer=summary_writer
    )

    return agent

if __name__ == '__main__':
    cli()

    