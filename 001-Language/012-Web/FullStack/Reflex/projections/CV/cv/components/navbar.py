import reflex as rx 


def navbar():
    """顶部导航栏组件"""
    
    return rx.hstack(
        rx.text("Yu Zhao", font_size="1.5rem", font_weight="bold", color="white"),
        rx.spacer(),
        rx.link("Home", href="#home", color="white", margin_x="1em"),
        rx.link("Publications", href="#publications", color="white", margin_x="1em"),
        rx.link("Projects", href="#projects", color="white", margin_x="1em"),
        rx.color_mode.button(position="bottom-right"),

        background_color="#4b2e83",  # 深紫色背景

        margin='0px',
        padding="1em 2em",
        width="100%", 
        align_items="center",

        # position="sticky",
        position="fixed",
        top="0",
        left="0",
        right="0",
        # bottom="1em",  # 避免遮挡内容
        z_index="1000",
    )