import torch

def greedy_decode(logits: torch.Tensor, charset, blank_idx: int):
    pred_indices = logits.argmax(dim=2)

    texts = []
    for seq in pred_indices:
        chars = []
        prev = None
        for idx in seq.tolist():
            if idx != prev:
                if idx != blank_idx:
                    chars.append(charset[idx])
            prev = idx
        texts.append("".join(chars))
    return texts


def edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def character_error_rate(preds, targets) -> float:
    total_dist, total_len = 0, 0
    for pred, target in zip(preds, targets):
        total_dist += edit_distance(pred, target)
        total_len += len(target)
    return total_dist / max(1, total_len)