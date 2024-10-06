def comp_outputs(a,b):
    output = ""
    a = a.strip("\n")
    b = b.strip("\n")
    split_a = set(a.rstrip().lstrip().split(" "))
    split_b = set(b.rstrip().lstrip().split(" "))
    diff = split_b.difference(split_a)
    if (diff):
        output += f"ERR: {len(diff)} mismatches found\n"
        output += ", ".join(diff)
    else:
        output = "Match"
    return output


cpu = ""
gpu = ""


print(comp_outputs(cpu, gpu))
