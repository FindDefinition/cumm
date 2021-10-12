import contextlib
import contextvars
from typing import Any, Optional

DEBUG_CONTEXT_VAR = contextvars.ContextVar("CudasimDebug")


class DebugContext:
    def __init__(self, enable: bool = True, debug_tx: int = 0):
        self.enable = enable
        self.debug_tx = debug_tx


def get_debug_context() -> Optional[DebugContext]:
    return DEBUG_CONTEXT_VAR.get(None)


def enable_debug():
    ctx = get_debug_context()
    if ctx is None:
        return False
    return ctx.enable


def debug_tx():
    ctx = get_debug_context()
    if ctx is None:
        return -1
    if not ctx.enable:
        return -1
    return ctx.debug_tx


@contextlib.contextmanager
def enter_debug_context(enable: bool = True, debug_tx: int = 0):
    ctx = DebugContext(enable, debug_tx)
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
