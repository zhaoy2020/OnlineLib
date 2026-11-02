import matplotlib 
import matplotlib.pyplot as plt 

from matplotlib_inline.backend_inline import set_matplotlib_formats

from IPython import display 
from loguru import logger 


journal_config = {
    # === Font ===
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    # 'font.family': 'serif',
    # 'font.serif': ['Times New Roman', 'SimSun', 'KaiTi'],

    # === Backend ===
    'svg.fonttype': 'none',                 # svg格式字体不转换为路径
    'pdf.fonttype': 42,                     # ai可编辑的字体格式
    'ps.fonttype': 42,                      # ai可编辑的字体格式
    'axes.unicode_minus': False,            # 解决负号显示问题
    'savefig.transparent': True,            # 背景是否透明

    # === Axes ===
    'axes.titlesize': 8,
    'axes.titlelocation': 'left',
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,

    # === Legend ===
    'legend.fontsize': 6,

    # === Figure ===
    'figure.figsize': (85/25.4, 85/(25.4*1.4)),
    'figure.constrained_layout.use': True,             # 自动调整子图参数，使之填充整个图像区域
}


def set_plt_default() -> None:
    '''
    Set default for plt.

    Demo:
    >>>set_plt_default()
    '''
    plt.rcdefaults()
    logger.info("Set default for plt.")


def set_plt_rcParams(format: str = "svg", **kswargs) -> None:
    '''Set configure for plt.
    Example:
    >>> set_plt_rcParams()
    '''

    # Update the default rc settings with the given dictionary of rc settings.
    plt.rcParams.update(journal_config)

    # Display with svg format rather than save status.
    # %config InlineBackend.figure_format = "svg" same as above in IPython
    set_matplotlib_formats(format)
    
    # Update with the given dictionary of rc settings.
    for k, v in kswargs.items():
        plt.rcParams[k] = v

    logger.info("Set configure for plt.")

    return None


def refresh_plot(fig:matplotlib.figure.Figure) -> None:
    '''Refresh show plot in jupyter.'''
    plt.close()                                  # close figure （推荐）
    display.display(fig)                         # 在jupyter中展示 （推荐）
    display.clear_output(wait= True)             # 等待 （必须）