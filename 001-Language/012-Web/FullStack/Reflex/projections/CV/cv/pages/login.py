import reflex as rx 
from ..state import AuthState


# @rx.page("/login")
def login():
    return rx.container(
        rx.heading("Login"),
        # rx.input(type="text", placeholder="Username", value=rx.bind(AuthState.username)),
        # rx.input(type="password", placeholder="Password", value=rx.bind(AuthState.password)),
        rx.button("Login", on_click=AuthState.login),
    )

