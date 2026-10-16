"""Minimal vecio-planner call with no dependencies beyond the binding."""

import vecio_planner as vp


def main() -> None:
    requests = [
        (0, 0, 8_192),
        (1, 8_192, 8_192),
        (2, 5_000_000, 8_192),
    ]
    cost = {
        "latency_per_request": 27.0,
        "latency_per_byte": 1.0e-5,
        "concurrent_parallelism": 16.0,
    }

    output = vp.plan(requests, "main_solution", **cost)
    metrics = vp.evaluate_plan(requests, output, **cost)

    print("merged requests:")
    for index, (offset, length) in enumerate(output["merged_requests"]):
        print(f"  [{index}] offset={offset} length={length}")
    print(f"assignments: {output['assignments']}")
    print(f"predicted latency: {metrics['predicted_latency_ms_t_ub']:.2f} ms")
    print(
        f"bytes: {metrics['useful_bytes']} useful + "
        f"{metrics['wasted_bytes']} wasted"
    )


if __name__ == "__main__":
    main()
