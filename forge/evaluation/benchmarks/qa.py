QA_BENCHMARK = [
    {
        "prompt": "问题：Transformer架构的核心机制是什么？\n答案：",
        "reference": "自注意力机制",
    },
    {
        "prompt": "问题：什么是梯度消失问题？\n答案：",
        "reference": "在深层神经网络中，梯度在反向传播过程中逐层衰减，导致靠近输入层的参数更新缓慢",
    },
    {
        "prompt": "问题：BERT和GPT的主要区别是什么？\n答案：",
        "reference": "BERT是双向编码器，GPT是单向自回归解码器",
    },
    {
        "prompt": "问题：什么是过拟合？如何解决？\n答案：",
        "reference": "过拟合是模型在训练集上表现好但泛化能力差。解决方法包括正则化、dropout、数据增强、早停等",
    },
    {
        "prompt": "问题：什么是Batch Normalization？\n答案：",
        "reference": "批归一化是对每个mini-batch的特征进行归一化处理，加速训练收敛并起到正则化效果",
    },
]


class QABenchmark:
    """Question answering accuracy benchmark."""

    def __init__(self):
        self.data = QA_BENCHMARK

    def evaluate(self, model, tokenizer, device: str = "cuda") -> dict:
        from ..harness import EvalHarness

        harness = EvalHarness()
        results = harness.evaluate_qa(model, tokenizer, self.data, device=device)
        return results

    def get_data(self) -> list[dict]:
        return self.data
