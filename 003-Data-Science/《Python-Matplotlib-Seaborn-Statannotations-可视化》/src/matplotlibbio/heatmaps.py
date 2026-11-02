from loguru import logger

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib import patheffects 
from matplotlib import colors
from matplotlib import lines


def heatmap(data: pd.DataFrame, fig: plt.Figure | None = None, ax: plt.Axes | None = None, ) -> plt.Axes:
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    hm = ax.imshow(data.values, cmap='coolwarm', norm=colors.Normalize(vmin=-1, vmax=1))
    fig.colorbar(hm, ax=ax, shrink=0.5)

    return ax


def cartoon_heatmap_demo_data() -> pd.DataFrame:
    '''Demo data for cartoon_heatmap
    Example:
    >>> data_df = cartoon_heatmap_demo_data()
    '''
    
    datas = np.random.random(12).reshape(3, 4)
    columns = ['A', 'B', 'C', 'D']
    index = ['X', 'Y', 'Z']
    data_df = pd.DataFrame(datas, columns=columns, index=index)

    return data_df


def cartoon_heatmap(data: pd.DataFrame, cmap: str = 'coolwarm', fig: plt.Figure | None = None, ax: plt.Axes | None = None) -> plt.Axes:
    '''Draw a cartoon-style heatmap with rounded rectangles and value annotations.
    Args:
        data: pd.DataFrame, the data to be visualized as a heatmap. The index and columns will be used for labeling.
        cmap: str, the name of the colormap to use for coloring the heatmap cells. Default is 'coolwarm'.
        fig: plt.Figure, optional, the figure to draw on. If None, uses the current figure.
        ax: plt.Axes, optional, the axes to draw on. If None, uses the current axes.
    Returns:
        plt.Axes, the axes with the drawn heatmap.
    Example:
        data = pd.DataFrame(np.random.random((5, 5)) * 2 - 1, columns=[f'Col{i}' for i in range(5)], index=[f'Row{i}' for i in range(5)])
        fig, ax = plt.subplots(figsize=(6, 6))
        cartoon_heatmap(data, fig=fig, ax=ax)    
    '''
    
    # cmap = plt.cm.coolwarm
    cmap = plt.get_cmap(cmap)
    norm = colors.Normalize(vmin=-1, vmax=1)
    
    if fig is None:
        fig = plt.gcf()

    if ax is None: 
        ax = plt.gca()

    for row_idx, row in enumerate(data.index[::-1]):
        for col_idx, col in enumerate(data.columns):
            value = data.loc[row, col]
            color = cmap(norm(value))
            rect = patches.FancyBboxPatch(
                xy=(col_idx, row_idx), # From the bottom-left corner
                width=0.9, height=0.9, 
                boxstyle='round,pad=0.02,rounding_size=0.2', 
                facecolor=color, 
                edgecolor='lightgray', 
                linewidth=0.5
            )
            ax.add_patch(rect)
            ax.text(
                x=col_idx + 0.45, 
                y=row_idx + 0.45, 
                s=f'{value:.2f}', 
                ha='center', va='center', 
                fontsize=6, color='black',
                # path_effects=[
                #     # patheffects.withStroke(linewidth=1, foreground='white'),                                  # Add a white stroke around the text for better visibility
                #     patheffects.withSimplePatchShadow(offset=(0.5, -0.5), shadow_rgbFace='white', alpha=0.8)    # Add a subtle shadow for better visibility
                # ]
            )
    ax.set_xlim(0, len(data.columns))
    ax.set_ylim(0, len(data.index))
    ax.set_xticks(np.arange(len(data.columns)) + 0.45)
    ax.set_yticks(np.arange(len(data.index)) + 0.45)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.index[::-1])
    ax.set_aspect('equal')

    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) # mappable for colorbar
    fig.colorbar(mappable=sm, ax=ax, shrink=0.5, use_gridspec=True)

    return ax


def mantel_heatmap_demo_data() -> tuple[pd.DataFrame]:
    '''Demo datas for mantel_heatmap
    Example:
    >>> data_df, correlation_df, p_value_df = mantel_heatmap_demo_data()
    '''

    N: int = 5
    factors: list = [f'factor_{i}' for i in range(1, N+1)]
    values: np.ndarray = np.random.rand(N*N).reshape(N, N)
    data_df: pd.DataFrame = pd.DataFrame(values, columns=factors, index=factors)

    # linewidth
    M: int = 5
    strains: list = [f'strain_{i}' for i in range(1, M+1)]
    correlations: np.ndarray = np.random.rand(N*M).reshape(N, M)
    correlations_df = pd.DataFrame(correlations, columns=strains, index=factors)

    # linecolor
    p_values: np.ndarray = np.random.rand(N, M)
    p_values_df = pd.DataFrame(p_values, columns=strains, index=factors)

    return (data_df, correlations_df, p_values_df)


def mantel_heatmap(
    heatmap_data: pd.DataFrame, 
    connection_width_data: pd.DataFrame,
    connection_color_data: pd.DataFrame,
    heatmap_cmap: str = 'coolwarm',
    connection_cmap: str = 'default',
    fig: plt.Figure | None = None, 
    ax: plt.Axes | None = None,
    cmap: str = 'coolwarm',
    connection_scale: float = 2.5
) -> plt.Axes:
    '''Draw a mantel-style heatmap with diagonal lines and value annotations.
    Args:
        heatmap_data: pd.DataFrame, 
        connection_width_data: pd.DataFrame,
        connection_color_data: pd.DataFrame,
        heatmap_cmap: str, the name of the colormap to use for coloring the heatmap cells. Default is 'coolwarm'.
        connection_cmap: str, the name of the colormap to use for coloring the connection lines. Default is 'coolwarm'.
        fig: plt.Figure, optional, the figure to draw on. If None, uses the current figure.
        ax: plt.Axes, optional, the axes to drawon.
    Returns:
        ax: plt.Axes
    Examples:
    >>> data_df, correlation_df, p_value_df = mantel_heatmap_demo_data()
    >>> ax = mantel_heatmap(heatmap_data=data_df, connection_width_data=correlation_df, connection_color_data=p_value_df, ax=ax)
    '''
    
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    # === color ===
    heatmap_cmap = plt.get_cmap(heatmap_cmap)
    heatmap_norm = colors.Normalize(vmin=-1, vmax=1)

    if connection_cmap == 'default':
        connection_cmap = colors.ListedColormap(colors=['#b2182b', '#ef8a62', '#d9d9d9'], name='default', N=3)
    else:
        connection_cmap = plt.get_cmap(connection_cmap)
    connection_norm = colors.BoundaryNorm(boundaries=[0, 0.01, 0.05, 1], ncolors=connection_cmap.N)

    # === Left bottom of Axes, Draw heatmap with heatmap_data ===
    for col_idx, col in enumerate(heatmap_data.columns):
        for row_idx, row in enumerate(heatmap_data.index[::-1]):
            if col_idx < (len(heatmap_data.index)-row_idx-1):
                value = heatmap_data.loc[row, col]
                color = heatmap_cmap(heatmap_norm(value))
                rect = patches.FancyBboxPatch(
                    xy=(col_idx, row_idx),                      # From the bottom-left corner
                    width=0.9, height=0.9, 
                    boxstyle='round,pad=0.02,rounding_size=0.2', 
                    facecolor=color, 
                    edgecolor='lightgray', 
                    linewidth=0.5
                )
                ax.add_patch(rect)
                ax.text(
                    x=col_idx + 0.45, 
                    y=row_idx + 0.45, 
                    s=f'{value:.2f}', 
                    ha='center', va='center', 
                    fontsize=6, color='black',
                )

            elif col_idx == (len(heatmap_data.index)-row_idx-1):
                circle = patches.Circle(
                    xy=(col_idx + 0.45, row_idx + 0.45), 
                    radius=0.2, 
                    facecolor='lightgray', 
                    edgecolor='lightgray', 
                    linewidth=0.5
                )
                ax.add_patch(circle)

            else:
                continue

    ax.set_xlim(0, len(heatmap_data.columns) + 1)
    ax.set_ylim(0, len(heatmap_data.index) + 1)
    ax.set_xticks(np.arange(len(heatmap_data.columns)) + 0.5)
    ax.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
    ax.set_yticklabels(heatmap_data.index[::-1])
    ax.set_aspect('equal')

    for spine in ax.spines.values():
        spine.set_visible(False)

    # === Right upper Axes, draw connections with connection_width_data and connection_color_data ===
    ## === Strian nodes ===
    curve_formula = lambda x: -1 * x + len(connection_width_data.index)*1.5
    strain_num = len(connection_width_data.columns)

    strain_X = np.linspace(strain_num/2, strain_num, strain_num)
    strain_positions = {
        strain_name: (x, curve_formula(x)) for strain_name, x in zip(connection_width_data.columns, strain_X)
    }

    for strain_name, (x, y) in strain_positions.items():
        strain_circle = patches.Circle(
            xy=(x, y), 
            radius=0.1, 
            facecolor='lightgray', 
            edgecolor='lightgray', 
            linewidth=0.5,
        )
        ax.add_patch(strain_circle)
        offset = 0.3
        ax.annotate(
            strain_name, 
            xy=(x, y), 
            xytext=(x + offset, y), 
            textcoords='data', 
            ha='left', va='center', 
            fontsize=6, color='black'
        )

    ## === Connection line ===
    factor_positions: dict = {
        factor_name: (x+0.5, len(connection_width_data.index)-x-0.5) for factor_name, x in zip(connection_width_data.index, range(len(connection_width_data.index))) 
    }
    for row in range(len(connection_width_data.index)):
        for col in range(len(connection_width_data.columns)):
            correlation_value = connection_width_data.iloc[row, col]
            p_value = connection_color_data.iloc[row, col]
            line = patches.FancyArrowPatch(
                posA=strain_positions[connection_width_data.columns[col]],
                posB=factor_positions[connection_width_data.index[row]],
                arrowstyle='-',
                linewidth=abs(correlation_value) * connection_scale,
                connectionstyle='arc3,rad=-0.1',
                color=connection_cmap(connection_norm(p_value)),
                alpha=0.7,
                zorder=0,
            )
            ax.add_patch(line)

    # === Add legend ===
    legend_pos_x: float = 1.05

    ## colorbar
    heatmap_cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=heatmap_norm), 
        ax=ax,
        shrink=0.3,
        use_gridspec=True, 
        orientation= 'horizontal',
        location='bottom',
        pad=0.02,
    )
    heatmap_cbar.ax.set_title('Heatmap value', fontsize=6)

    ## Connection width
    r_handles = [
        lines.Line2D([0],[0], color='black', lw=connection_scale*0.2, label='0.2'),
        lines.Line2D([0],[0], color='black', lw=connection_scale*0.6, label='0.6'),
        lines.Line2D([0],[0], color='black', lw=connection_scale*1, label='1')
    ]
    legend_r = ax.legend(
        handles=r_handles,
        title=f"|$r$| value",
        loc='upper left',
        bbox_to_anchor=(legend_pos_x ,0.99),
        frameon=False,
        fontsize=6,
        title_fontsize=6,
    )
    ax.add_artist(legend_r)

    # Connection color
    p_handles = [
        lines.Line2D([0], [0], color=connection_cmap(connection_norm(0.005)), lw=2, label='[0.00, 0.01)'),
        lines.Line2D([0], [0], color=connection_cmap(connection_norm(0.03)), lw=2, label='[0.01, 0.05)'),
        lines.Line2D([0], [0], color=connection_cmap(connection_norm(0.75)), lw=2, label='[0.05, 1.00]'),
    ]
    lelgend_p = ax.legend(
        handles=p_handles, 
        title=f'$p$-value',
        loc='upper left', 
        bbox_to_anchor=(legend_pos_x, 0.49), 
        fontsize=6,
        title_fontsize=6,
        frameon=False,
    )

    return ax