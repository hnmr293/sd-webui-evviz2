# =======================================================================================
from scripts.lib.utils import ensure_install
ensure_install('plotly')
ensure_install('pandas')
# =======================================================================================

import sys
import traceback
from typing import List, Tuple, Dict, Callable, Union

import numpy as np
import torch
from torch import Tensor, nn
import plotly.graph_objects as go
import gradio as gr

from modules import script_callbacks
from modules import shared
from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWordsBase as CLIP

NAME = 'EvViz2'

def generate_embeddings(
    te: CLIP,
    prompt: str,
    padding: Union[str,int]
) -> Tuple[List[Tuple[int,str]], Tensor, Tensor]:
    
    if hasattr(te.wrapped, 'tokenizer'):
        # v1
        tokenizer = te.wrapped.tokenizer
        token_to_id: Callable[[str],int] = lambda t: tokenizer._convert_token_to_id(t)
        id_to_token: Callable[[int],str] = lambda t: tokenizer.convert_ids_to_tokens(t)
        ids_to_tokens: Callable[[List[int]],List[str]] = lambda ts: tokenizer.convert_ids_to_tokens(ts)
    else:
        # v2
        import open_clip
        tokenizer = open_clip.tokenizer._tokenizer
        token_to_id: Callable[[str],int] = lambda t: tokenizer.encoder[t]
        id_to_token: Callable[[int],str] = lambda t: tokenizer.decoder[t]
        ids_to_tokens: Callable[[List[int]],List[str]] = lambda ts: [tokenizer.decoder[t] for t in ts]
    
    if isinstance(padding, str):
        padding = token_to_id(padding)
    
    tokens = te.tokenize([prompt])[0]
    #vocab = tokenizer.get_vocab()
    #vocab = { token_num: id_to_token(token_num) for idx, token_num in enumerate(tokens) }
    
    prompts = [prompt]
    for idx in range(len(tokens)):
        ts = tokens[:idx] + [padding] + tokens[idx+1:]
        words = ''.join(ids_to_tokens(ts)).replace('</w>', ' ')
        prompts.append(words)
    
    embeddings = te(prompts).cpu()
    return (
        [(token_num, id_to_token(token_num)) for token_num in tokens],
        embeddings[0, 1:-1, :],
        embeddings[1:, 1:-1, :]
    )

def run(prompt: str, padding: Union[str,int,None], skip_comma: bool, gl: bool):
    if padding is None:
        padding = 0
    elif isinstance(padding, str):
        if len(padding) == 0:
            padding = 0
        else:
            try:
                padding = int(padding)
            except:
                pass
    
    te: CLIP = shared.sd_model.cond_stage_model # type: ignore
    tokens, v, vs = generate_embeddings(te, prompt, padding)
    # v := (75, 768) or (75, 1024)
    v_max = torch.max(v)
    
    # main plot
    fig = go.Figure()
    xs = list(range(v.shape[1]))
    y_count = 0
    for idx, (token_num, token) in enumerate(tokens):
        if skip_comma and token == ',</w>':
            continue
        base_y = y_count * v_max/2
        vals = v[idx] + base_y
        fig.add_trace(
            [go.Scatter, go.Scattergl][gl](
                x=xs,
                y=vals,
                mode='lines+markers',
                name=f'token {idx}: {token} ({token_num})',
                marker=dict(
                    size=6,
                    symbol='circle',
                    color='rgba(0,0,0,0)',
                    line=dict(
                        color='rgba(255,64,64,1)',
                        width=1,
                    ),
                ),
                line=dict(
                    color='rgba(255,64,64,0.5)',
                    width=2,
                ),
                hovertemplate='%{x}<br>%{customdata[0]}',
                customdata=v[idx].unsqueeze(1),
            )
        )
        y_count += 1
    
    fig.update_layout(
        yaxis=dict(showticklabels=False),
        legend=dict(xanchor="left", yanchor="top"),
    )
    
    # self-correlation
    fig2 = go.Figure()
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
    for idx, (token_num, token) in enumerate(tokens):
        if skip_comma and token == ',</w>':
            continue
        w = vs[idx]
        dv = v - w # (75, 768)
        dv_norm = torch.linalg.vector_norm(dv, dim=1) # (75,)
        dv_norm[idx] = 0.0
        correls.append(dv_norm)
        indices.append(idx)
    
    cs = torch.gather(
        torch.vstack(correls), 
        dim=1,
        index=torch.LongTensor([indices] * len(correls)),
    )
    
    fig2.add_trace(
        go.Heatmap(
                z=cs,
                #x=[ t.replace('</w>', '') for i, t in tokens ],
                #y=[ t.replace('</w>', '') for i, t in tokens ],
                colorscale=[[0, 'rgb(64,64,255)'], [1, 'rgb(255,0,0)']],
                hovertemplate='%{y} -> %{x}<br>%{z} <extra></extra>',
                hoverlabel=dict(font_family='monospace'),
                xgap=1,
                ygap=1,
            )
    )
    
    ticks = [ tokens[index][1].replace('</w>', '') for index in indices ]
    fig2.update_layout(
        xaxis=dict(
            side='top',
            tickvals=list(range(cs.shape[1])),
            ticktext=ticks,
            tickfont=dict(family='monospace'),
        ),
        yaxis=dict(
            autorange='reversed',
            #scaleanchor='x',
            tickvals=list(range(cs.shape[0])),
            ticktext=ticks,
            tickfont=dict(family='monospace'),
        )
    )
    
    return fig, fig2

def add_tab():
    def wrap(fn):
        def f(*args, **kwargs):
            v, e = None, ''
            try:
                v = fn(*args, **kwargs)
            except Exception as ex:
                e = traceback.format_exc()
                print(e, file=sys.stderr)
            if isinstance(v, tuple):
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
        button = gr.Button(variant='primary')
        graph = gr.Plot()
        graph2 = gr.Plot()
    
        button.click(fn=wrap(run), inputs=[prompt, padding, skip, gl], outputs=[graph, graph2, error])
    
    return [(ui, NAME, NAME.lower())]

script_callbacks.on_ui_tabs(add_tab)
