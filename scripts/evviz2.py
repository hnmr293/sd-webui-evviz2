# =======================================================================================
from scripts.evviz2lib.utils import ensure_install
ensure_install('plotly')
# =======================================================================================

import sys
import traceback
from dataclasses import dataclass
import colorsys
from typing import List, Tuple, Dict, Callable, Union

import numpy as np
import torch
from torch import Tensor, nn
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import gradio as gr

from modules import script_callbacks
from modules import shared
from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWordsBase as CLIP

from scripts.evviz2lib.sdhook import SDModel, each_unet_attn_layers
from scripts.evviz2lib.sdhook import GeneralConditioner as CLIP_SDXL

NAME = 'EvViz2'

@dataclass
class Token:
    id: int
    token: str

class Context:
    def __init__(self, context: Tensor, token_count: int):
        self._context = context
        self.token_count = token_count
    
    @property
    def context(self):
        return self._context[:self.token_count, 1:-1, :]
    
    @property
    def base(self):
        return self._context[0, 1:-1, :]
    
    @property
    def padded(self):
        return self._context[1:self.token_count+1, 1:-1, :]
    
    @property
    def device(self):
        return self._context.device
    
    @property
    def dtype(self):
        return self._context.dtype
    
    def to(self, device, dtype, inplace=False):
        if inplace:
            self._context = self._context.to(device, dtype)
            return self
        else:
            return Context(self._context.to(device, dtype), self.token_count)


def clip_type(te):
    if isinstance(te, CLIP):
        if hasattr(te.wrapped, 'tokenizer'):
            return 'v1'
        else:
            return 'v2'
    elif hasattr(te, 'embedders'):
        return 'xl'
    else:
        raise RuntimeError(f'unknown text encoder: {te.__class__.__name__}')


def generate_embeddings(
    te: Union[CLIP,CLIP_SDXL],
    prompt: str,
    padding: Union[str,int]
) -> Tuple[List[Token], Context]:
    ty = clip_type(te)
    print(f'[{NAME}] clip type: {ty}')
    if ty in ('v1',):
        tokenizer = te.wrapped.tokenizer
        token_to_id: Callable[[str],int] = lambda t: tokenizer._convert_token_to_id(t)
        id_to_token: Callable[[int],str] = lambda t: tokenizer.convert_ids_to_tokens(t)
        ids_to_tokens: Callable[[List[int]],List[str]] = lambda ts: tokenizer.convert_ids_to_tokens(ts)
    elif ty in ('v2','xl'):
        import open_clip
        tokenizer = open_clip.tokenizer._tokenizer
        token_to_id: Callable[[str],int] = lambda t: tokenizer.encoder[t]
        id_to_token: Callable[[int],str] = lambda t: tokenizer.decoder[t]
        ids_to_tokens: Callable[[List[int]],List[str]] = lambda ts: [tokenizer.decoder[t] for t in ts]
    else:
        raise RuntimeError(f'unknown type of text encoder: {ty} [{te.__class__.__name__}]')
    
    if isinstance(padding, str):
        padding = token_to_id(padding)
    
    print(f'[{NAME}] Padding token: {id_to_token(padding)} ({padding})', file=sys.stderr)
    
    tokens = te.tokenize([prompt])[0]
    #vocab = tokenizer.get_vocab()
    #vocab = { token_num: id_to_token(token_num) for idx, token_num in enumerate(tokens) }
    
    prompts = [prompt]
    for idx in range(len(tokens)):
        ts = tokens[:idx] + [padding] + tokens[idx+1:]
        words = ''.join(ids_to_tokens(ts)).replace('</w>', ' ')
        prompts.append(words)
    
    if ty in ('v1','v2'):
        embeddings = te(prompts)
    else:
        assert ty == 'xl'
        a1 = te.embedders[0](prompts)    # (n, 77,  768)
        a2 = te.embedders[1](prompts)[0] # (n, 77, 1280)
        embeddings = torch.cat((a1, a2), dim=-1) # (n, 77, 2048)
    
    embeddings = embeddings.cpu()
    return (
        [Token(token_num, id_to_token(token_num)) for token_num in tokens],
        Context(embeddings, len(tokens))
    )

def build_main_graph(
    title: str,
    context_: Context,
    tokens: List[Token],
    skip_comma: bool,
    use_gl: bool,
    force_float: bool,
):
    fig = go.Figure()
    
    context = context_.base
    xs = list(range(context.shape[1]))
    v_max = torch.max(context)
    y_count = 0
    
    hlsa_0 = torch.FloatTensor([0.0, 0.625, 1.0])
    hlsa_1 = torch.FloatTensor([-1/3, 0.5, 1.0])
    for pos, tt in enumerate(tokens):
        if skip_comma and tt.token == ',</w>':
            continue
        
        color = torch.lerp(hlsa_0, hlsa_1, pos/len(tokens))
        r, g, b = [int(255*x) for x in colorsys.hls_to_rgb(*[x.item() for x in color])]
        base_y = y_count * v_max/2
        vals = context[pos] + base_y
        
        if force_float:
            vals = vals.to('cpu', dtype=torch.float)
        else:
            vals = vals.to('cpu')
        
        fig.add_trace(
            [go.Scatter, go.Scattergl][use_gl](
                x=xs,
                y=vals,
                mode='lines+markers',
                name=f'token {pos}: {tt.token} ({tt.id})',
                marker=dict(
                    size=6,
                    symbol='circle',
                    color='rgba(0,0,0,0)',
                    line=dict(
                        color=f'rgba({r},{g},{b},1)',
                        width=1,
                    ),
                ),
                line=dict(
                    color=f'rgba({r},{g},{b},0.5)',
                    width=2,
                ),
                hovertemplate='%{x}<br>%{customdata[0]}',
                customdata=context[pos].unsqueeze(1).to('cpu'),
            )
        )
        y_count += 1
    
    fig.update_layout(
        title=dict(text=title),
        yaxis=dict(showticklabels=False),
        legend=dict(xanchor="left", yanchor="top"),
    )
    
    return fig
    

def create_correl_map(
    context: Context,
    tokens: List[Token],
    skip_comma: bool,
    ignore_self_correl: bool,
    use_gl: bool,
    force_float: bool,
):
    # self-correlation
    
    """
          a   cute   girl
    a     *
    cute  **
    girl  
    
    *  = emb(a cute girl) - emb($ cute girl)
    ** = emb(a cute girl) - emb(a $ girl)
         ~~~~~~~~~~~~~~~~     ~~~~~~~~~~~
         `- v                 `- w = vs[idx]
    """
    
    correls = []
    indices = []
    for pos, tt in enumerate(tokens):
        if skip_comma and tt.token == ',</w>':
            continue
        dv = context.base - context.padded[pos]
        dv_norm = torch.linalg.vector_norm(dv, dim=1) # (75,)
        if ignore_self_correl:
            #dv_norm[pos] = 0.0
            dv_norm[pos] = np.nan
        correls.append(dv_norm)
        indices.append(pos)
    
    cs = torch.gather(
        torch.vstack(correls), 
        dim=1,
        index=torch.LongTensor([indices] * len(correls)).to(context.device),
    )
    
    if force_float:
        cs = cs.to('cpu', dtype=torch.float)
    else:
        cs = cs.to('cpu')
    
    map = go.Heatmap(
        z=cs,
        colorscale=[[0, 'rgb(64,64,255)'], [1, 'rgb(255,0,0)']],
        hovertemplate='%{y} -> %{x}<br>%{z} <extra></extra>',
        hoverlabel=dict(font_family='monospace'),
        xgap=1,
        ygap=1,
    )
    
    return map, cs, indices


def build_correl_graph(
    titles: List[str],
    contexts: List[Context],
    tokens: List[Token],
    skip_comma: bool,
    ignore_self_correl: bool,
    use_gl: bool,
    force_float: bool,
):
    if len(contexts) == 0:
        return
    
    assert len(titles) == len(contexts)
    assert len(set([ctx.token_count for ctx in contexts])) == 1, set([ctx.token_count for ctx in contexts])
    
    fig = make_subplots(
        rows=len(contexts),
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.01,
    )
    
    for row, context in enumerate(contexts, start=1):
        map, cs, indices = create_correl_map(context, tokens, skip_comma, ignore_self_correl, use_gl=use_gl, force_float=force_float)
        ticks = [ tokens[index].token.replace('</w>', '') for index in indices ]
        fig.add_trace(map, row=row, col=1)
        fig.update_xaxes(
            row=row, col=1,
            tickvals=list(range(cs.shape[1])),
            ticktext=ticks,
        )
        fig.update_yaxes(
            row=row, col=1,
            autorange='reversed',
            tickvals=list(range(cs.shape[0])),
            ticktext=ticks,
        )
    
    bgcolor = ','.join(str(x) for x in (255, 255, 255, 1))
    
    fig.update_layout(
        xaxis=dict(
            side='top',
            tickfont=dict(family='monospace'),
            showgrid=False,
            showline=False,
        ),
        yaxis=dict(
            #scaleanchor='x',
            tickfont=dict(family='monospace'),
            showgrid=False,
            showline=False,
        ),
        height=300*len(titles),
        plot_bgcolor=f'rgba({bgcolor})',
    )
    
    for row in range(len(contexts)):
        fig.update_traces(
            row=row, col=1,
            colorbar=dict(
                yanchor='top',
                y=1-row/len(contexts),
                len=1/len(contexts),
            )
        )
    
    return fig
    

def run(
    prompt: str,
    padding: Union[str,int,None],
    skip_comma: bool,
    ignore_self_correl: bool,
    force_float: bool,
    to_k: bool,
    to_v: bool,
    gl: bool
):
    if padding is None:
        padding = '_</w>'
    elif isinstance(padding, str):
        if len(padding) == 0:
            padding = '_</w>'
        else:
            try:
                padding = int(padding)
            except:
                pass
    
    sd_model = SDModel(shared.sd_model)
    te = sd_model.clip
    unet = sd_model.unet
    
    tokens, context = generate_embeddings(te, prompt, padding)
    # v := (75, 768) or (75, 1024)
    
    # main plot
    fig = build_main_graph('Embedding Vector', context, tokens, skip_comma, use_gl=gl, force_float=True)
    
    # CLIP
    titles = [te.wrapped.__class__.__name__]
    contexts = [context]
    
    if to_k:
        for name, mod in each_unet_attn_layers(unet):
            if 'xattn' in name:
                wk = mod.to_k.weight
                dtype = wk.dtype
                if force_float:
                    dtype = torch.float
                context.to(wk.device, dtype, inplace=True)
                k = mod.to_k(context._context)
                titles.append(name + '.to_k')
                contexts.append(Context(k, len(tokens)))
        
    if to_v:
        for name, mod in each_unet_attn_layers(unet):
            if 'xattn' in name:
                wk = mod.to_k.weight
                dtype = wk.dtype
                if force_float:
                    dtype = torch.float
                context.to(wk.device, dtype, inplace=True)
                v = mod.to_v(context._context)
                titles.append(name + '.to_v')
                contexts.append(Context(v, len(tokens)))
        
    fig2 = build_correl_graph(titles, contexts, tokens, skip_comma, ignore_self_correl, use_gl=gl, force_float=True)
    
    return fig, fig2

def add_tab():
    def wrap(fn, values: int = 1):
        def f(*args, **kwargs):
            v, e = None, ''
            try:
                with torch.inference_mode():
                    v = fn(*args, **kwargs)
            except Exception:
                ex = traceback.format_exc()
                print(ex, file=sys.stderr)
                e = str(ex).replace('\n', '<br/>')
            if 1 < values:
                if v is None:
                    v = [None] * values
                return [*v, e]
            else:
                return [v, e]
        return f
    
    with gr.Blocks(analytics_enabled=False) as ui:
        error = gr.HTML(elem_id=f'{NAME.lower()}-error')
        gl = gr.Checkbox(value=True, label='Use WebGL for plotting')
        prompt = gr.TextArea(label='Prompt')
        with gr.Row():
            padding = gr.Textbox(label='Padding token (ID or single token)')
            skip = gr.Checkbox(value=True, label='Skip commas (,)')
            ignore_self_correl = gr.Checkbox(value=False, label='Ignore self-correlation')
            to_k = gr.Checkbox(value=True, label='to_k (xattn)')
            to_v = gr.Checkbox(value=True, label='to_v (xattn)')
            force_float = gr.Checkbox(value=False, label='Force convert half to float (for some platforms)')
        button = gr.Button(variant='primary')
        close = gr.Button(value='Close')
        graph = gr.Plot()
        graph2 = gr.Plot()
    
        def close_fn():
            return None, None
        
        button.click(fn=wrap(run, 2), inputs=[prompt, padding, skip, ignore_self_correl, force_float, to_k, to_v, gl], outputs=[graph, graph2, error])
        close.click(fn=close_fn, inputs=[], outputs=[graph, graph2])
    
    return [(ui, NAME, NAME.lower())]

script_callbacks.on_ui_tabs(add_tab)
