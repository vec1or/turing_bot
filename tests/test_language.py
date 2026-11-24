import unittest
import numpy as np
from turing_bot.language import Language, Pieces


class LanguageTests(unittest.TestCase):
    def test_visible_word_boundary_and_roundtrip(self):
        vocab=Pieces.learn(["В саду играет музыка.", "Где мой синий зонт?"], 100)
        numbers=vocab.encode("В саду музыка.")
        self.assertEqual(vocab.decode(numbers),"В саду музыка.")

    def test_trained_transformer_and_numpy_match(self):
        try:
            import torch
            from training.train_language import Net
        except ImportError:
            self.skipTest("Для сверки реализаций нужен PyTorch из среды обучения.")
        runtime=Language.load()
        model=Net(runtime.config).eval()
        model.load_state_dict({k:torch.from_numpy(v) for k,v in runtime.w.items()})
        ids=runtime.pieces.prompt("О какой книге идёт речь?",["книга"])
        with torch.no_grad():
            reference=model(torch.tensor([ids]))[0].numpy()
        states=None
        actual=[]
        for i in ids:
            logits,states=runtime.step(i,states)
            actual.append(logits)
        np.testing.assert_allclose(actual,reference,rtol=1e-3,atol=2e-4)
