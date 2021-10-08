import contextlib
import contextvars
from typing import Any, Optional

DEBUG_CONTEXT_VAR = contextvars.ContextVar("CudasimDebug")


class DebugContext:
    def __init__(self, enable: bool = True):
        self.enable = enable


def get_debug_context() -> Optional[DebugContext]:
    return DEBUG_CONTEXT_VAR.get(None)


def enable_debug():
    ctx = get_debug_context()
    if ctx is None:
        return False
    return ctx.enable


@contextlib.contextmanager
def enter_debug_context(enable: bool = True):
    ctx = DebugContext(enable)
    token = DEBUG_CONTEXT_VAR.set(ctx)
    yield ctx
    DEBUG_CONTEXT_VAR.reset(token)


def debug_print(*args: Any):
    ctx = get_debug_context()
    if ctx is None:
        return print(*args)
    else:
        if ctx.enable:
            return print(*args)
    return
