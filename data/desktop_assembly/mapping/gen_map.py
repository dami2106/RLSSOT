x = {
"a":"Pick up chip",
"b":"Place chip on motherboard",
"c":"Close cover",
"d":"Pick up screw and screwdriver",
"e":"Tighten screw",
"f":"Plug stick in",
"g":"Pick up fan",
"h":"Place fan on motherboard",
"i":"Tighten screw 1",
"j":"Tighten screw 2",
"k":"Tighten screw 3",
"l":"Tighten screw 4",
"m":"Put screwdriver down",
"n":"Connect wire to motherboard",
"o":"Pick up RAM",
"p":"Install RAM",
"q":"Pick up HDD",
"r":"Install HDD",
"s":"Connect wire 1 to HDD",
"t":"Connect wire 2 to HDD",
"u":"Pick up lid",
"v":"Close lid",
"w":"Background"
}

actions = x.values()

with open('mapping.txt', 'w') as f:
    for i, act in zip(range(len(actions)), actions):
        f.write(f'{i} {act.replace(" ", "_")}\n')
