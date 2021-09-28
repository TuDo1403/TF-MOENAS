import logging

class CallbackHandler:
    def __init__(self, callbacks=None, summary_writer=None) -> None:
        self.summary_writer = summary_writer
        self.callbacks = callbacks if callbacks else []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.msg = 'gen {}, n_eval {}: {}'
        self.agent = None

    def begin_fit(self, agent, **kwargs):
        self.agent = agent
        msgs = []
        for callback in self.callbacks:
            msg = callback._begin_fit(
                agent=agent, 
                callbacks=self.callbacks, 
                summary_writer=self.summary_writer, 
                **kwargs
            )
            if msg:
                msgs += [msg]
        if len(msgs) > 0:
            self.logger.info(self.msg.format(
                self.agent.model.n_gen,
                self.agent.model.evaluator.n_eval,
                str(msgs)
            ))

    def after_fit(self, **kwargs):
        msgs = []
        for callback in self.callbacks:
            msg = callback._after_fit(**kwargs)
            if msg:
                msgs += [msg]
        if len(msgs) > 0:
            self.logger.info(self.msg.format(
                self.agent.model.n_gen,
                self.agent.model.evaluator.n_eval,
                str(msgs)
            ))

    def begin_next(self, **kwargs):
        msgs = []
        for callback in self.callbacks:
            msg = callback._begin_next(**kwargs)
            if msg:
                msgs += [msg]
        if len(msgs) > 0:
            self.logger.info(self.msg.format(
                self.agent.model.n_gen,
                self.agent.model.evaluator.n_eval,
                str(msgs)
            ))

    def after_next(self, **kwargs):
        msgs = []
        for callback in self.callbacks:
            msg = callback._after_next(**kwargs)
            if msg:
                msgs += [msg]
        if len(msgs) > 0:
            self.logger.info(self.msg.format(
                self.agent.model.n_gen,
                self.agent.model.evaluator.n_eval,
                str(msgs)
            ))