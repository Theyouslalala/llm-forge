from ..metrics import compute_bleu, compute_rouge_l


TEXT_GEN_BENCHMARK = [
    {
        "prompt": "请用一句话解释什么是机器学习：",
        "reference": "机器学习是人工智能的一个分支，通过算法让计算机从数据中自动学习规律和模式，无需显式编程即可做出预测和决策。",
    },
    {
        "prompt": "请简要介绍Python语言的特点：",
        "reference": "Python是一种高级编程语言，具有语法简洁、易于学习、跨平台、拥有丰富的第三方库、支持面向对象和函数式编程等特点。",
    },
    {
        "prompt": "请解释什么是神经网络：",
        "reference": "神经网络是一种受生物神经系统启发的计算模型，由多层相互连接的人工神经元组成，通过调整连接权重来学习数据中的复杂模式。",
    },
    {
        "prompt": "请用一句话描述什么是云计算：",
        "reference": "云计算是通过互联网提供计算资源和服务的技术，用户按需使用远程服务器的存储、计算和应用能力，无需拥有物理硬件。",
    },
    {
        "prompt": "请简要说明什么是区块链技术：",
        "reference": "区块链是一种去中心化的分布式账本技术，通过密码学和共识机制确保数据的不可篡改性和透明性，广泛应用于加密货币和智能合约。",
    },
]


class TextGenBenchmark:
    """Text generation quality benchmark."""

    def __init__(self):
        self.data = TEXT_GEN_BENCHMARK

    def evaluate(self, model, tokenizer, device: str = "cuda") -> dict:
        from ..harness import EvalHarness

        harness = EvalHarness()
        results = harness.evaluate_generation(
            model, tokenizer, self.data,
            task_name="text_gen_benchmark",
            device=device,
            metrics=["bleu", "rouge_l"],
        )
        return results

    def get_data(self) -> list[dict]:
        return self.data
