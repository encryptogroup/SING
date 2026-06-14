def base_assignment_is(allowed, metadata):
    if metadata["mode"] not in ["random", "perturb", "neighborloader", "op-type"]:
        return True

    return metadata["base-assignment"] in allowed

def only_noise(metadata):
    if not metadata["mode"] == "random":
        return True

    return metadata["noise-prob"] == "1"


assignment_filters = {
    "no_filter": lambda metadata: True,

    "no_all": lambda metadata: metadata["mode"] not in ["all-b", "all-y"],
    "all": lambda metadata: metadata["mode"] in ["all-b", "all-y"],

    "silph_and_random": lambda metadata: metadata["mode"] in ["silph", "random"] and base_assignment_is(["silph"], metadata),
    "silph_and_perturb": lambda metadata: metadata["mode"] in ["silph", "perturb"] and base_assignment_is(["silph"], metadata),
    "silph_and_neighborloader": lambda metadata: metadata["mode"] in ["silph", "neighborloader"] and base_assignment_is(["silph"], metadata),
    "spn": lambda metadata: metadata["mode"] in ["silph", "perturb", "neighborloader"] and base_assignment_is(["silph"], metadata),
    "span": lambda metadata: metadata["mode"] in ["silph", "perturb", "all-b", "all-y", "neighborloader"] and base_assignment_is(["silph"], metadata),
    "spano": lambda metadata: metadata["mode"] in ["silph", "perturb", "all-b", "all-y", "neighborloader", "op-type"] and base_assignment_is(["silph"], metadata),

    "ao": lambda metadata: metadata["mode"] in ["all-b", "all-y", "op-type"] and base_assignment_is(["all-b", "all-y"], metadata),
    "ap": lambda metadata: metadata["mode"] in ["all-b", "all-y", "perturb"] and base_assignment_is(["all-b", "all-y"], metadata),
    "ano": lambda metadata: metadata["mode"] in ["all-b", "all-y", "neighborloader", "op-type"] and base_assignment_is(["all-b", "all-y"], metadata),
    "apno": lambda metadata: metadata["mode"] in ["all-b", "all-y", "perturb", "neighborloader", "op-type"] and base_assignment_is(["all-b", "all-y"], metadata),
    "apnor": lambda metadata: metadata["mode"] in ["all-b", "all-y", "perturb", "neighborloader", "op-type"] and base_assignment_is(["all-b", "all-y"], metadata) or metadata["mode"] == "random" and only_noise(metadata),
}
