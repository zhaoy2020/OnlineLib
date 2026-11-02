import reflex as rx 

from typing import Callable

from .components.navbar import navbar 
from .components.menu import menu
from .components.footer import footer


def template(page: Callable[[], rx.Component]) -> rx.Component:
    """The main template for the CV project, as a decorator."""

    return rx.vstack(
        navbar(),
        page(),
        footer(),

        spacing="1",
        padding="0",
        align='center',
    )

