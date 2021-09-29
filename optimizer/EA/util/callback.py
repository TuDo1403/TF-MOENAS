import torch

import logging

import pickle5 as pickle

class CallbackBase:
    def __init__(self, verbose=True) -> None:
        super().__init__()
        self.algorithm = None
        self.callbacks = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.summary_writer = None
        self.verbose = verbose
        self.agent = None

    def _begin_fit(self, agent, callbacks, summary_writer=None, **kwargs):
        self.algorithm = agent.model
        self.agent = agent
        self.callbacks = callbacks
        self.summary_writer = summary_writer
    
    def _after_fit(self, **kwargs):
        pass

    def _begin_next(self, **kwargs):
        pass

    def _after_next(self, **kwargs):
        pass

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

import numpy as np

from itertools import combinations

import time

import re

class NonDominatedProgress(CallbackBase):
    def __init__(self, 
                plot_freq=1,
                plot_pf=True, 
                labels=None,
                **kwargs):
        super().__init__(**kwargs)
        self.plot_freq = plot_freq
        self.labels = labels
        self.plot_info = None
        self.path = None
        self.plot_pf = plot_pf

    def _begin_fit(self, **kwargs):
        super()._begin_fit(**kwargs)

        n_obj = self.algorithm.problem.n_obj  
        if not self.labels:
            self.labels = [r'$f_{}(x)$'.format((i+1)) for i in range(n_obj)]

        n_obj = list(combinations(range(n_obj), r=2))
        ax_labels = list(combinations(self.labels, r=2))
        pf = self.algorithm.problem.pareto_front()
        if pf is None:
            pf = [None] * len(ax_labels)
        else:
            pf = list(combinations(pf.T.tolist(), r=2))
        pf = [None] * len(ax_labels)
        
        self.plot_info = [n_obj, ax_labels, pf]
        # self.path = os.path.join(
        #     self.agent.cfg.gif_dir,
        #     '[{}][{}][{}-{}]-G{:0>3d}.jpg'.format(
        #         self.algorithm.__class__.__name__,
        #         self.algorithm.__class__.__name__,
        #         self.algorithm.n_gen
        #     )
        # )
         

    def _begin_next(self, **kwargs):
        if self.algorithm.n_gen is not None and self.algorithm.n_gen % self.plot_freq == 0:
            f_pop = self.algorithm.pop.get('F'); f_opt = self.algorithm.opt.get('F')
            for i, (obj_pair, labels, data) in enumerate(zip(*self.plot_info)):
                fig = self.__plot_figure(f_pop, f_opt, obj_pair, labels, data)
                if self.summary_writer:
                    self.summary_writer.add_figure(
                        tag='fig/' + re.sub(r'(?u)[^-\w.]', '', '{}-{}'.format(*labels)),
                        figure=fig,
                        global_step=self.algorithm.n_gen
                    )
                    self.summary_writer.close()
                fig.savefig(
                    os.path.join(
                    self.agent.cfg.gif_dir,
                    re.sub(r'(?u)[^-\w.]', '', '[{}-{}]-G{:0>3d}.jpg'.format(
                        *labels,
                        self.algorithm.n_gen
                    ))
                )
                )
                if self.verbose:
                    plt.show()
                plt.close(fig)

    def __plot_figure(self, 
                      f_pop, 
                      f_opt, 
                      obj_pair, 
                      labels, 
                      data):
        fig, ax = plt.subplots()
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
        if data:
            ax.plot(*data, label='pareto front', color='red')
                
        X = f_pop[:, obj_pair[0]]; Y = f_pop[:, obj_pair[1]]
        X_opt = f_opt[:, obj_pair[0]]; Y_opt = f_opt[:, obj_pair[1]]
        # lim = ax.get_xlim(), ax.get_ylim()
        ax.scatter(X, Y, marker='o', color='green', facecolors='none', label='gen: {}'.format(self.algorithm.n_gen))
        ax.plot(X_opt[np.argsort(X_opt)], Y_opt[np.argsort(X_opt)], 'g--')
        ax.legend(loc='best')
        ax.set_title('Objective Space')
        ax.grid(True, linestyle='--')
        fig.tight_layout()
        return fig

from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume

# from pymoo.performance_indicator.igd import IGD
# from pymoo.performance_indicator.hv import Hypervolume

import pandas as pd

class PerformanceMonitor(CallbackBase):
    def __init__(self, 
                 metric, 
                 normalize=True,
                 from_archive=False,
                 convert_to_pf_space=False,
                 topk=1,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.convert_to_pf_space = convert_to_pf_space
        self.monitor = None
        self.metric = metric
        self.normalize = normalize
        self.from_archive = from_archive
        self.topk = topk
        self.top_lst = []
        self.current_score = None
        self.current_time = 0
        self.current_gen = 0
        self.data = []

    def __repr__(self) -> str:
        info = {
            'metric': self.metric,
            'current_val': self.current_score,
            'top_k': self.top_lst
        }
        return str(info)

    
    def _after_next(self, F, reverse, **kwargs):
        score = self.monitor.do(F)
        # score = self.monitor.calc(F)
        self.current_score = score
        self.top_lst += [score]
        self.top_lst = list(set(self.top_lst))
        self.top_lst = sorted(self.top_lst, reverse=reverse)
        if len(self.top_lst) > self.topk:
            self.top_lst = self.top_lst[:self.topk]
        if self.verbose:
            if score in self.top_lst:
                msg = \
                    '{}={:.3f} (best={:.3f})'.format(
                        self.metric.lower(),
                        score, 
                        self.top_lst[0]
                    )
            else:
                msg = \
                    '{} was not in top {}'.format(
                        self.metric.lower(), 
                        self.topk
                    )
        if self.algorithm.n_gen == 1:
            self.current_time = 0
        elif self.current_gen != self.algorithm.n_gen:
            self.current_gen = self.algorithm.n_gen
            try:
                self.current_time += sum(self.algorithm.problem.history['runtime'][self.current_gen-1])
            except:
                pass
        
        if self.summary_writer:
            
            self.summary_writer.add_scalar(
                tag='metric/{}'.format(self.metric),
                scalar_value=score,
                global_step=self.algorithm.evaluator.n_eval,
                walltime=self.current_time
            )
            self.summary_writer.close()
        self.data += [[self.current_time, self.algorithm.n_gen, self.algorithm.evaluator.n_eval, score]]
        df = pd.DataFrame(self.data, columns=['walltime', 'n_gen', 'n_eval', self.metric])
        df.to_csv(os.path.join(self.agent.config.out_dir, '{}.csv'.format(self.metric)))
        return msg

    def _calc_F(self):
        if self.convert_to_pf_space:
            if self.from_archive:
                X = self.algorithm.problem.elitist_archive.get('X')
            else:
                X = self.algorithm.opt.get('X')
            F = self.algorithm.problem._convert_to_pf_space(X)
        else:
            if self.from_archive:
                F = self.algorithm.problem.elitist_archive.get('F')
            else:
                F = self.algorithm.opt.get('F')
        return F


class IGDMonitor(PerformanceMonitor):
    def __init__(self,**kwargs) -> None:
        super().__init__('IGD', **kwargs)

    def _begin_fit(self,**kwargs):
        super()._begin_fit(**kwargs)
        pf = self.algorithm.problem.pareto_front()
        self.monitor = eval(self.metric)(pf=pf, zero_to_one=self.normalize)

    def _after_next(self, **kwargs):
        F = self._calc_F()
        return super()._after_next(F=F, reverse=False, **kwargs)


class HyperVolumeMonitor(PerformanceMonitor):
    def __init__(self, ref_point=[100, 100], **kwargs) -> None:
        super().__init__('Hypervolume', **kwargs)
        self.ref_point = ref_point
    
    def _begin_fit(self,**kwargs):
        super()._begin_fit(**kwargs)
        

    def _after_next(self, **kwargs):
        F = self._calc_F()
        approx_ideal = F.min(axis=0)
        approx_nadir = F.max(axis=0)
        self.monitor = eval(self.metric)(
            ref_point=self.ref_point, 
            norm_ref_point=False, 
            zero_to_one=self.normalize,
            ideal=approx_ideal,
            nadir=approx_nadir
        )
        return super()._after_next(F=F, reverse=True, **kwargs)

            
class TimeLogger(CallbackBase):
    def __init__(self, log_freq=1, **kwargs):
        super().__init__(**kwargs)
        self.log_freq = log_freq
        self.start = None
        self.history = []
        self.msg = 'time={:.4f}s (avg={:.4f}s)'

    def _begin_next(self, **kwargs):
        self.start = time.time()

    def _after_next(self, **kwargs):
        if self.algorithm.n_gen % self.log_freq == 0:
            end = time.time() - self.start
            self.history += [end]
            msg = self.msg.format(
                end,
                sum(self.history)/len(self.history)
            )
            return msg

import os

import copy

class CheckpointSaver(CallbackBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.filepath = '[{algorithm}_{problem}] G-{:0>3d}.ckp'
    
    def _after_next(self, **kwargs):
        ckp = {
            'cfg': self.agent.cfg,
            'model': copy.deepcopy(self.algorithm),
        }

        filepath = self.filepath.format(
            self.algorithm.n_gen,
            algorithm=self.algorithm.__class__.__name__,
            problem=self.algorithm.problem.__class__.__name__
        )
        torch.save(
            obj=ckp,
            f=os.path.join(
                self.agent.cfg.checkpoint_dir,
                filepath
            ),
            pickle_module=pickle,
            pickle_protocol=5
        )






# class CheckpointSaver(CallbackBase):
#     def __init__(self, ckp_dir, monitor=None, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.ckp_dir = ckp_dir
#         self.metric = monitor
#         self.monitor = None
#         self.found_monitor = False

#     def _begin_next(self, **kwargs):
#         if self.metric and not self.found_monitor:
#             for cb in self.callbacks:
#                 if self.metric in repr(cb):
#                     self.found_monitor = True
#                     self.monitor = cb
#         if not self.found_monitor:
#             self.logger.warn('Metric for saving checkpoint not found!')
    
#     def _after_next(self, **kwargs):
#         if self.metric and self.found_monitor:

    