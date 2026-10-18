import torch

from ctf.mappo.trainer import MAPPOTrainer

CFG = {"run_name": "smoke", "n_envs": 2, "T": 16, "save_every": 2, "anneal_epochs": 2,
       "env": {"size": 6, "max_steps": 10}}


def test_random_mode_and_resume(tmp_path):
    tr = MAPPOTrainer(CFG, root=tmp_path)
    assert list(tr.learners) == ["blue"]
    tr.train(2)
    assert (tmp_path / "smoke" / "ckpt" / "epoch_000002.pt").exists()
    assert tr.beta() == 0.0  # beta llegó a 0 tras anneal_epochs
    tr2 = MAPPOTrainer(CFG, root=tmp_path)
    tr2.resume()
    assert tr2.epoch == 2


def test_selfplay_two_separate_networks(tmp_path):
    cfg = {**CFG, "run_name": "sp", "opponent": "selfplay", "snapshot_every": 1, "save_every": 1,
           "p_past": 0.5, "env": {"size": 8, "n_per_team": 2, "max_steps": 12, "n_boxes": 1, "n_walls": 1}}
    tr = MAPPOTrainer(cfg, root=tmp_path)
    blue, red = tr.learners["blue"], tr.learners["red"]
    assert blue.actor is not red.actor and blue.opt is not red.opt
    tr.train(2)
    # los pesos de cada equipo son independientes
    wb = torch.cat([p.flatten() for p in blue.actor.parameters()])
    wr = torch.cat([p.flatten() for p in red.actor.parameters()])
    assert not torch.allclose(wb, wr)
    blob = torch.load(tmp_path / "sp" / "ckpt" / "latest.pt", weights_only=False)
    assert {"actor_blue", "actor_red", "critic_blue", "critic_red"} <= set(blob["models"])
    assert {"opt_blue", "opt_red"} <= set(blob["optim"])
    assert all(len(tr.league[t].members) == 3 for t in ("blue", "red"))  # random + 2 copias
    tr2 = MAPPOTrainer(cfg, root=tmp_path)
    tr2.resume()
    assert tr2.epoch == 2 and all(len(tr2.league[t].members) == 3 for t in ("blue", "red"))
