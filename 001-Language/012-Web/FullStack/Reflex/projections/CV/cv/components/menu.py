import reflex as rx 


def menu() -> rx.Component:
    '''The menu component.'''

    return rx.hstack(
        rx.link("Home", href="/"),
        rx.link("Login", href="/login"),
        spacing="1",
    )