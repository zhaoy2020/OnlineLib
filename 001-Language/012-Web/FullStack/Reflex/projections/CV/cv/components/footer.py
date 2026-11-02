import reflex as rx 


def footer():
    """底部组件"""
    
    return rx.hstack(
        rx.text("© 2024 Yu Zhao. All rights reserved."),
        rx.spacer(),
        rx.link("GitHub", href="https://github.com/yuzhao1990"),
    )