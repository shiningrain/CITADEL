
import torch


def main():
    cpu = torch.device("cpu")
    cuda = torch.device("cuda:0")

    tensorboard_profile_dir = "/tmp/tensorboard_profile/"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=2, active=1, repeat=1),
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_profile_dir),
    ) as profile:

        batch_size = 1
        num_feat = 1500  # Increase this to make the cuda sync in clip_grad_norm_ more dramatic.
        model = torch.nn.Linear(
            in_features=num_feat, out_features=num_feat, bias=False, device=cuda
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        with torch.profiler.record_function("init data"):
            # Random input data, and fixed regression target.
            x = torch.randn(batch_size, num_feat, device=cuda)
            target = torch.full((num_feat,), 100.0, device=cuda)

        for _ in range(3):
            with torch.profiler.record_function("forward + loss"):
                y = model(x)
                l1_loss = (y - target).abs().mean()

            with torch.profiler.record_function("backward"):
                optimizer.zero_grad()
                l1_loss.backward()

            with torch.profiler.record_function("clip grad"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type="inf")

            optimizer.step()

            profile.step()

    print("Saved profile to", tensorboard_profile_dir)


if __name__ == "__main__":
    main()
