import numpy as np
import torch
import time
import math

torch.set_printoptions(8)

def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]

        Input: Tensor
        Output: Tensor
    """
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x ** 3)))
    # return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))



def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    x_exp = torch.exp(x - torch.max(x))

    return x_exp / x_exp.sum(dim=0, keepdim=True)

import torch

def layer_norm(x, g_b, eps: float = 1e-5):
    """
    Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
    Input:
        x: Tensor
        g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
    Output: Tensor
    """
    # 计算均值和标准差
    mean = x.mean(dim=-1, keepdim=True)  # 计算最后一个维度的均值
    std = x.std(dim=-1, keepdim=True)    # 计算最后一个维度的标准差

    # 进行层归一化
    x_normalized = (x - mean) / (std + eps)  # 归一化

    # 使用 gamma 和 beta 进行缩放和偏移
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])  # 加载 gamma 和 beta
    return g * x_normalized + b                           # 应用缩放和偏移



def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer
        Input:
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w = torch.tensor(w_b['w'], dtype=torch.float32)  # 确保 w 是张量
    b = torch.tensor(w_b['b'], dtype=torch.float32)  # 确保 b 是张量
    return torch.matmul(x, w) + b


def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` and `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input:
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    # 提取第一个和第二个线性层的权重和偏置
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']

    # 第一个线性变换
    x = linear(x, w_b1)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # 应用激活函数 GELU
    x = gelu(x)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # 第二个线性变换
    x = linear(x, w_b2)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x  # 输出最终结果



def attention(q, k, v, mask=None):
    """
    Task: use torch API to implement attention computation according to formula (1) of the following paper,
          where d_k accounts for the last dimension of `k`.
    Paper: https://arxiv.org/abs/1706.03762
    Input:
        q: Tensor
        k: Tensor
        v: Tensor
        mask: Tensor
    Output: Tensor
    """
    dk = q.size(-1)
    # 计算缩放的点积注意力分数
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)  # [n_head, n_seq, n_seq]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 计算注意力权重
    attn_weights = torch.softmax(scores, dim=-1)  # [n_head, n_seq, n_seq]

    # 计算注意力输出
    output = torch.matmul(attn_weights, v)  # [n_head, n_seq, head_dim]
    return output

def mha(x, attn, n_head):     # [n_seq, n_embd] -> [n_seq, n_embd]
    
    """
    Task: Complete the code of the multi-head attention.

    Input:
        x: Tensor
        attn: dictionary that load from GPT-2 weights. c_attn and c_proj are the params of two linear layers.
        n_head: number of heads
    Output: Tensor after multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']

    # 线性映射得到 Q、K、V
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3 * n_embd]

    n_seq = x.size(0)
    n_embd = c_proj['w'].shape[0]  # 嵌入维度
    head_dim = n_embd // n_head  # 每个头的维度

    # 正确地拆分 Q、K、V
    q, k, v = x.chunk(3, dim=-1)  # 等分为 q, k, v，每个形状：[n_seq, n_embd]

    # 分割成多头并调整形状
    q = q.view(n_seq, n_head, head_dim).transpose(0, 1)  # [n_head, n_seq, head_dim]
    k = k.view(n_seq, n_head, head_dim).transpose(0, 1)
    v = v.view(n_seq, n_head, head_dim).transpose(0, 1)

    # 创建因果掩码
    causal_mask = torch.tril(torch.ones(n_seq, n_seq)).to(x.device)  # [n_seq, n_seq]
    causal_mask = causal_mask.unsqueeze(0)  # [1, n_seq, n_seq]

    # 调用 attention 函数
    attn_output = attention(q, k, v, mask=causal_mask)  # [n_head, n_seq, head_dim]

    # 调整形状并合并所有头
    attn_output = attn_output.transpose(0, 1).contiguous().view(n_seq, -1)  # [n_seq, n_embd]

    # 通过输出线性层
    x = linear(attn_output, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x


def transformer_block(x, block, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']

    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]

def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm  #显示进度条的工具

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate:]  # only return generated ids


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    '''
    encoder为编码器，hparams为超参数，params为模型参数
    '''

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)