import torch

from ctf.checkpoint import RunStore


def test_roundtrip(tmp_path):
    s = RunStore("t", root=tmp_path)
    net = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(net.parameters())
    s.save(10, 1000, {"actor": net}, {"opt": opt}, {"winrate": 0.5})
    s.save(20, 2000, {"actor": net}, {"opt": opt})
    assert [c["epoch"] for c in s.list()] == [10, 20]
    net2 = torch.nn.Linear(3, 2)
    assert s.restore({"actor": net2}, {"opt": opt}, which=10) == (10, 1000)
    assert torch.equal(net.weight, net2.weight)
