from abc import ABC, abstractmethod

import logging

from .util.callback_handler import CallbackHandler


class AgentBase(ABC):
    def __init__(self, config, callbacks, summary_writer=None, **kwargs):
        self.config = config
        self.logger = logging.getLogger(name=self.__class__.__name__)

        self.callback_handler = CallbackHandler(callbacks, summary_writer)

        self.model = None

    
    ### Public Method ###
    def solve(self, **kwargs):
        try:
            self.run(**kwargs)
        except KeyboardInterrupt:
            self.logger.info("Interrupted. You have entered CTRL+C...")
        except Exception as e:
            self.logger.error(e, exc_info=True)

    def run(self, **kwargs):
        self._initialize(**kwargs)
        self.callback_handler.begin_fit(agent=self, **kwargs)   
        while self.model.has_next():
            self.callback_handler.begin_next(**kwargs)
            self.model.next()
            self.callback_handler.after_next(**kwargs)
        self._finalize(**kwargs)

    ### Public Method ###

    ### Virtual Methods ###
    def _load_checkpoint(self, api, cmd=None, **kwargs):
        if cmd:
            ckp = eval(cmd)
        else:
            ckp = api.load(**kwargs)
        return ckp
    ### Virtual Methods ###

    @abstractmethod
    def _initialize(self, **kwargs):
        raise NotImplementedError

