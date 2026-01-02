import json
import queue
import threading
from pathlib import Path

import torch


def _save_worker(save_q: "queue.Queue", output_dir: Path):
    """
    Minimal replica of the writer behavior: write N sentence_*.pt + metadata.json.
    """
    while True:
        item = save_q.get()
        if item is None:
            save_q.task_done()
            break
        try:
            sample_id = item["sample_id"]
            token_ids = item["token_ids"]
            positions = item["positions"]
            activations = item["activations"]
            label = float(item["label"])
            token_texts = item["token_texts"]
            meta = item["metadata"]

            sample_dir = output_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            sentence_data = []
            for pos_idx, token_pos in enumerate(positions):
                pos_activation = activations[:, pos_idx:pos_idx + 1, :]
                pt_path = sample_dir / f"sentence_{pos_idx}.pt"
                torch.save((token_ids, pos_activation), pt_path)
                sentence_data.append(
                    {
                        "sentence_index": pos_idx,
                        "sentence": token_texts[pos_idx],
                        "p_reward_hacks": label,
                        "token_position": int(token_pos),
                    }
                )

            with open(sample_dir / "metadata.json", "w") as f:
                json.dump(
                    {
                        "sentence_data": sentence_data,
                        "sample_type": "baseline",
                        "labeling_method": "smeared_binary",
                        "source_file": meta.get("source_file"),
                        "sentence_idx": meta.get("sentence_idx"),
                        "rollout_idx": meta.get("rollout_idx"),
                        "is_hacking": meta.get("is_hacking"),
                        "num_layers": int(activations.shape[0]),
                        "hidden_dim": int(activations.shape[2]),
                    },
                    f,
                    indent=2,
                )
        finally:
            save_q.task_done()


def test_batched_activation_indexing():
    """
    Ensure our intended indexing works for (L, B, S, H) tensors.
    """
    L, B, S, H = 2, 3, 8, 4
    all_acts = torch.arange(L * B * S * H).reshape(L, B, S, H)

    positions = [1, 6]
    acts = all_acts[:, 2, positions, :]  # sample index 2
    assert acts.shape == (L, len(positions), H)

    # Spot-check one value matches the original tensor
    assert acts[1, 0, 3].item() == all_acts[1, 2, 1, 3].item()


def test_async_writer_writes_expected_format(tmp_path: Path):
    """
    Validate the on-disk structure matches quick_load expectations:
      sample_dir/sentence_{i}.pt contains (token_ids, activations)
      sample_dir/metadata.json has sentence_data with p_reward_hacks
    """
    save_q: "queue.Queue" = queue.Queue()
    t = threading.Thread(target=_save_worker, args=(save_q, tmp_path), daemon=True)
    t.start()

    token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    positions = [1, 3]
    activations = torch.randn(3, len(positions), 5)  # (num_layers, num_positions, hidden)

    save_q.put(
        {
            "sample_id": "baseline_test_s0_r0",
            "token_ids": token_ids,
            "positions": positions,
            "activations": activations,
            "label": 1.0,
            "token_texts": ["A", "B"],
            "metadata": {"source_file": "x.json", "sentence_idx": 0, "rollout_idx": 0, "is_hacking": True},
        }
    )

    save_q.put(None)
    save_q.join()
    t.join(timeout=5)

    sample_dir = tmp_path / "baseline_test_s0_r0"
    assert (sample_dir / "metadata.json").exists()
    assert (sample_dir / "sentence_0.pt").exists()
    assert (sample_dir / "sentence_1.pt").exists()

    loaded_token_ids, loaded_act0 = torch.load(sample_dir / "sentence_0.pt", map_location="cpu")
    assert torch.equal(loaded_token_ids, token_ids)
    assert loaded_act0.shape == (3, 1, 5)

    meta = json.loads((sample_dir / "metadata.json").read_text())
    assert meta["sentence_data"][0]["p_reward_hacks"] == 1.0

