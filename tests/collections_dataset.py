from kpcn_denoiser.dataset import collect_clean_noisy_samples

res = collect_clean_noisy_samples(
                    root="d:\\YandexDisk\\TGB_train_dataset\\",
                    noise_levels = ("high", "low", "verylow", "aggressive", "extreme", "mid",),
                    subsets = ("anim",),
                    layer = "chars",
                    aovs=("rgba",),
                    ext=".exr",
                    n_first_samples=10,
                    n_first_frames=2
                )
print(res)