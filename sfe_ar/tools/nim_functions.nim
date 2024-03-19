import nimpy
import tables

proc nim_subgraph(graph: PyObject, head, tail: string): PyObject {.exportpy.} =

    let 
        cl = pyImport("collections")
        subgraph = cl.`defaultdict`()
        nodes: seq[string]
        el: string

    subgraph[1] = cl.`defaultdict`()
    subgraph[2] = cl.`defaultdict`()

    for key in graph[head].keys():
        nodes = graph[head][key]
        for el in nodes:
            subgraph[1][el] = el


    return subgraph