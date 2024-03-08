#include <stdexcept>
#include <vector>
#include <queue>


template<class GRAPH, class VERTEX_PRIORITY_ITERATOR, class COMPONENT_LABEL_ITERATOR>
void
growDecomposition(
    const GRAPH& graph,
    VERTEX_PRIORITY_ITERATOR vertexPrioritiesBegin,
    VERTEX_PRIORITY_ITERATOR vertexPrioritiesEnd,
    size_t numberOfSeedVertices,
    COMPONENT_LABEL_ITERATOR componentLabelsBegin,
    COMPONENT_LABEL_ITERATOR componentLabelsEnd,
    size_t seed,
    bool separator_line = true
) {
    typedef typename COMPONENT_LABEL_ITERATOR::value_type ComponentLabel;

    // assert that the size of the graph, vertexPriorities and componentLabels are consistent
    if(std::distance(vertexPrioritiesBegin, vertexPrioritiesEnd) != graph.numberOfVertices()) {
        throw std::runtime_error("std::distance(vertexPrioritiesBegin, vertexPrioritiesEnd) != graph.numberOfVertices()");
    }
    if(std::distance(componentLabelsBegin, componentLabelsEnd) != graph.numberOfVertices()) {
        throw std::runtime_error("std::distance(componentLabelsBegin, componentLabelsEnd) != graph.numberOfVertices()");
    }

    if(numberOfSeedVertices <= 0) {
        throw std::runtime_error("Need at least one seed vertex");
    }

    // start with all vertices in component 0
    std::fill(componentLabelsBegin, componentLabelsEnd, static_cast<ComponentLabel>(0));

    // choose random seeds
    std::default_random_engine randomEngine;
    randomEngine.seed(seed);
    std::uniform_int_distribution<ComponentLabel> distribution(0, graph.numberOfVertices());

    auto cmp = [&vertexPrioritiesBegin](size_t const v, size_t const w) { return vertexPrioritiesBegin[v] < vertexPrioritiesBegin[w]; };
    typedef std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> VertexQueue;
    std::vector<VertexQueue> queues(numberOfSeedVertices, VertexQueue(cmp));
    for(ComponentLabel label = 1; label <= numberOfSeedVertices; ) {
        auto & queue = queues[label - 1];
        ComponentLabel seedVertex = distribution(randomEngine);
        
        // assert that the seedVertex is not already a seed of another component
        if (componentLabelsBegin[seedVertex] != 0){
            continue;
        }
        
        // assert that none of the neighbors of the seedVertex are the seed of another component
        if (separator_line)
        {
            bool neighbor_in_other_comp = false;
            for (auto neighbor = graph.verticesFromVertexBegin(seedVertex); neighbor < graph.verticesFromVertexEnd(seedVertex); neighbor++){
                if (componentLabelsBegin[*neighbor] != 0){
                    neighbor_in_other_comp = true;
                    break;
                }
            }
            if (neighbor_in_other_comp){
                continue;
            }
        }
        
        // set the label of the seedVertex to the current label and add it to the corresponding queue
        componentLabelsBegin[seedVertex] = label;
        queue.push(seedVertex);
        ++label;
    }

    // grow the components until all queues are empty
    for(;;) {
        bool allQueuesEmpty = true;
        for(ComponentLabel label = 1; label <= numberOfSeedVertices; ++label) {
            auto & queue = queues[label - 1];

            if(!queue.empty()) {
                size_t const vertex = queue.top();
                queue.pop();
                // iterate over all neighbors of the vertex. 
                for(auto n = graph.verticesFromVertexBegin(vertex); n != graph.verticesFromVertexEnd(vertex); ++n) {
                    // continue if n is already labeled
                    if (componentLabelsBegin[*n] != 0){
                        continue;
                    }
                    // if the neighbor n has a neighbor nn that is in a different component than the component of vertex,
                    // then n needs to remain in component 0, i.e. the separator. Otherwise the two components would be joined.
                    if (separator_line)
                    {
                        bool isNeighborOfDifferentComponent = false;
                        for(auto nn = graph.verticesFromVertexBegin(*n); nn != graph.verticesFromVertexEnd(*n); ++nn) {
                            if(componentLabelsBegin[*nn] == 0) {
                                continue;
                            }
                            if(componentLabelsBegin[*nn] == label) {
                                continue;
                            }
                            isNeighborOfDifferentComponent = true;
                            break;
                        }
                        if (isNeighborOfDifferentComponent){
                            continue;
                        }
                    }
                    // set the label of the neighbor n to the current label and add n to the queue
                    componentLabelsBegin[*n] = label;
                    queue.push(*n);
                }
            }
            if(!queue.empty()) {
                allQueuesEmpty = false;
            }
        }
        if(allQueuesEmpty) {
            break;
        }
    }
}
