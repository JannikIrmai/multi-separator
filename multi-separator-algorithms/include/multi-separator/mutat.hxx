#include <andres/graph/grid-graph.hxx>
#include <andres/partition.hxx>
#include <vector>
#include <array>
#include <queue>
#include <set>
#include <limits>
#include <map>

#include <chrono>
typedef std::chrono::high_resolution_clock::time_point TimePoint;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;

/*
Mutual attraction algorithm: 
- All vertices are repulsive (i.e. want to be in separator) and all interaction are attractive (i.e. endpoints should not be separated)

- Pseudo code of algorithm:
    Iterate over all vertices and interaction with ascending absolute values
        if the current element is a vertex
            if the vertex is part of a mutat component or removing the vertex results in a mutat component being separated: 
                continue
            else:
                add the vertex to the separator
        if the current element is an interaction:
            if the endpoints are in the same mutat component:
                continue
            if the endpoints are not connected wrt. to the current separator:
                continue
            else:
                join the mutat components of the endpoints of the interactions

- The bottlenecks of the algorithm:
    - checking if removing a vertex results in a mutat component being separated!
    - checking if the endpoints of an interaction are separated wrt. to the current separator (should not be a problem)
*/

namespace multi_separator
{


template<const char D>
class GridMutat
{
public:
    typedef andres::graph::GridGraph<D> GridGraph;

    GridMutat(const GridGraph& graph, const std::vector<std::array<int, D>>& offsets)
    :
        graph(graph),
        offsets(offsets),
        mutat_constraints(graph.numberOfVertices()),
        labels(graph.numberOfVertices(), 1),
        num_comp(1)
    {}

    GridMutat(const std::array<size_t, D>& shape, const std::vector<std::array<int, D>>& offsets)
    :
        graph(shape),
        offsets(offsets),
        mutat_constraints(graph.numberOfVertices()),
        labels(graph.numberOfVertices(), 1),
        num_comp(1)
    {}
    
    template<typename CONTAINER>
    void run_container(const CONTAINER& sorted_indices)
    {
        run(sorted_indices.data(), sorted_indices.data() + sorted_indices.size());
    }

    template<typename ITER>
    void run(ITER begin, ITER end)
    {
        TimePoint t_0 = Clock::now();
        Duration d;
        TimePoint t;
        for (; begin != end; ++begin)
        {
            if (*begin < graph.numberOfVertices())
            {
                t = Clock::now();
                add_vertex_to_separator(*begin);
                Duration d = Clock::now() - t;
                t_add_vertex += d.count();
            }
            else
            {
                TimePoint t = Clock::now();
                add_mutat(*begin - graph.numberOfVertices());
                Duration d = Clock::now() - t;
                t_add_mutat += d.count();
            }
        }
        d = Clock::now() - t_0;
        t_add_mutat += d.count();
        std::cout << "t_add_vertex: " << std::to_string(t_add_vertex) << std::endl;
        std::cout << "t_add_mutat:  " << std::to_string(t_add_mutat) << std::endl;
    }

    void reset()
    {
        std::fill(mutat_constraints.begin(), mutat_constraints.end(), std::set<size_t>());
        std::fill(labels.begin(), labels.end(), 1);
        num_comp = 1;
    }

    const std::vector<size_t>& vertex_labels() const
    {
        return labels;
    }

    std::vector<size_t> separator() const
    {
        std::vector<size_t> separator(labels.size(), 0);
        for (size_t i = 0; i < labels.size(); ++i)
            if (labels[i] == 0)
                separator[i] = 1;
        return separator;
    }

    void make_labels_consecutive()
    {
        std::map<size_t, size_t> label_map;
        for (auto label : labels)
            if (label != 0)
                label_map[label] = 0;
        num_comp = 1;
        for (auto& lab : label_map)
            if (lab.second == 0)
                lab.second = num_comp++;
        for (size_t i = 0; i < labels.size(); ++i)
            if (labels[i] != 0)
                labels[i] = label_map[labels[i]];
    }


private:
    
    GridGraph graph;
    std::vector<std::array<int, D>> offsets;
    std::vector<std::set<size_t>> mutat_constraints;
    std::vector<size_t> labels;
    size_t num_comp;

    double t_total = {0};
    double t_add_mutat = {0};
    double t_add_vertex = {0};

    std::tuple<bool, size_t, size_t> findEdge(size_t e) const
    {   
        size_t offset_idx = e / graph.numberOfVertices();
        size_t source = e % graph.numberOfVertices();
        
        std::array<size_t, D> source_coords;
        graph.vertex(source, source_coords);
        auto target_coords = source_coords;
        for (size_t d = 0; d < D; ++d)
        {
            target_coords[d] += offsets[offset_idx][d]; 
            if (((offsets[offset_idx][d] < 0) && (source_coords[d] < -offsets[offset_idx][d]) || (target_coords[d] >= graph.shape(d))))
                return {false, 0, 0};
        }
        size_t target = graph.vertex(target_coords);
        return {true, source, target};
    }

    // This method labels the edge e as a mutual attraction edge if both end-vertices of the edge have the same label
    void add_mutat(size_t e)
    {
        auto edge = findEdge(e);
        if (!std::get<0>(edge))
            return;
        size_t u = std::get<1>(edge);
        size_t v = std::get<2>(edge);
        if (labels[u] == 0)
            return;
        if (labels[u] == labels[v])
        {
            mutat_constraints[u].insert(v);
            mutat_constraints[v].insert(u);
        }

        // TODO: Utilize a union find data structure that keeps track of the mutat components:
        //  E.g.: If there are two mutat constraints ab and bc then the mutat constraint ac does not 
        //  need to be added as it is already implied by ab and bc.
    }

    // This method adds vertex v to the separator if this does not cause a mutat edge to be separated
    void add_vertex_to_separator(size_t s)
    {
        // This method checks if there is a mutat edge from the vertex v to any neighbor u whose label is not 
        // the same as the label of the vertex v or as the new_label
        auto check_mutat = [this] (size_t v, size_t new_label)
        {
            for (auto& u : mutat_constraints[v])
                if ((labels[u] != labels[v]) && (labels[u] != new_label))
                    return true;
            return false;
        };

        // add s to the separator (and potentially remove it again later)
        size_t label_s = labels[s];
        labels[s] = 0;

        // check if there is a mutat edge from s to any vertex, if yes, s cannot be added to the separator
        if (check_mutat(s, 0))
        {
            labels[s] = label_s;
            return;
        }

        // get all neighbors of s that are not in the separator
        std::vector<size_t> neighbors;
        for (auto it = graph.verticesFromVertexBegin(s); it != graph.verticesFromVertexEnd(s); ++it)
            if (labels[*it] > 0)
                neighbors.push_back(*it);

        // if there is at most one neighboring vertex, s can be added to the separator
        if (neighbors.size() <= 1) 
            return;

        // setup a bfs from all neighbors to check if s is a separating vertex
        std::vector<std::queue<size_t>> queues(neighbors.size());  // one queue for each bfs
        std::vector<size_t> new_labels(neighbors.size());  // label to mark in the labels array which vertices have been visited by each bfs
        for (size_t i = 0; i < neighbors.size(); ++i)
        {
            new_labels[i] = std::numeric_limits<size_t>::max() - i;
            labels[neighbors[i]] = new_labels[i];
            queues[i].push(neighbors[i]);
        }

        auto relabel = [this, &check_mutat] (size_t start, size_t new_label, bool check = false)
        {
            size_t old_label = labels[start];
            if (check && check_mutat(start, new_label))
                return true;

            // bfs from start to set all labels to new_label
            if (labels[start] == new_label)
                return false;
            
            std::queue<size_t> queue;
            queue.push(start);
            labels[start] = new_label;

            while (queue.size() > 0)
            {
                size_t w = queue.front();
                queue.pop();
                // iterate over all neighbors of v and add them to the queue
                for(auto it = graph.verticesFromVertexBegin(w); it != graph.verticesFromVertexEnd(w); ++it)
                {
                    if (labels[*it] == old_label)
                    {
                        if (check && check_mutat(*it, new_label))
                            return true;

                        labels[*it] = new_label;
                        queue.push(*it);
                    }
                }
            }
            return false;
        };

        size_t empty_queues = 0;  // count the number of empty queues
        // start bfs
        while (empty_queues < neighbors.size() - 1)
        {
            for (size_t i = 0; i < neighbors.size(); ++i)
            {
                if (queues[i].size() == 0)
                    continue;
                
                size_t v = queues[i].front();
                queues[i].pop();
                // iterate over all neighbors of v and add them to the queue
                for(auto it = graph.verticesFromVertexBegin(v); it != graph.verticesFromVertexEnd(v); ++it)
                {
                    // if *it is in the separator, do nothing
                    if (labels[*it] == 0)
                        continue;
                    // if *it was already visited by the current bfs, do nothing
                    else if (labels[*it] == new_labels[i])
                        continue;
                    // if *it is still labeled with label_s, then proceed with normal bfs
                    else if (labels[*it] == label_s)
                    {
                        queues[i].push(*it);
                        labels[*it] = new_labels[i];
                        
                    }
                    // otherwise, the label of *it is equal to another new_label. This means that
                    // another bfs was found. The found bfs is added to the current bfs and then the
                    // respective queue is cleared.
                    else
                    {
                        // get the index j of the found bfs
                        size_t j = 0;
                        for (; j < neighbors.size(); ++j)
                            if (new_labels[j] == labels[*it])
                                break;
                        
                        // relabel all vertices labels new_labels[j] to new_labels[i]
                        // TODO: It would be better if this was also done in the interleaved fashion.
                        //  With the current approach the component corresponding the current bfs 
                        //  can become the size of (deg(g) - 1)/deg(g) of the component of label_s.
                        // NOTE: With the current implementation the labels are not consecutive, anyways because
                        //  it is possible that all vertices from one component are being added to the separator.
                        relabel(*it, new_labels[i], false);
                        // clear queue j into queue i
                        while (queues[j].size() > 0)
                        {
                            queues[i].push(queues[j].front());
                            queues[j].pop();
                        }
                        ++empty_queues;
                    }
                }

                // check if the current bfs just finished
                if (queues[i].size() == 0)
                {
                    // check for violated mutat constraints
                    if(relabel(neighbors[i], ++num_comp, true))
                    {   
                        relabel(neighbors[i], new_labels[i], false); // undo the re-labeling
                        goto FOUND_MUTAT_VIOLATION;
                    }

                    // increase the empty queues counter
                    ++empty_queues;
                }

                if (empty_queues >= neighbors.size() - 1)
                    break;
            }
        }
        

        // For the non-empty queue: undo the corresponding bfs by 
        // changing all labels back to the original label label_s
        for (size_t i = 0; i < neighbors.size(); ++i)
            if (queues[i].size() > 0)
                relabel(neighbors[i], label_s, false);
        return;

        FOUND_MUTAT_VIOLATION:
        {
            labels[s] = label_s;
            for (size_t i = 0; i < neighbors.size(); ++i)
                relabel(neighbors[i], label_s, false);
        }
    }
};



template<typename GRAPH, typename INTERACTION_GRAPH>
class Mutat
{
public:

    Mutat(const GRAPH& graph, const INTERACTION_GRAPH& interaction_graph)
    :
        graph(graph),
        interaction_graph(interaction_graph),
        mutat_edges(interaction_graph.numberOfEdges()),
        labels(graph.numberOfVertices(), 1),
        num_comp(1)
    {}

    template<typename ITER>
    void run(ITER begin, ITER end)
    {
        for (; begin != end; ++begin)
        {
            if (*begin < graph.numberOfVertices())
                add_vertex_to_separator(*begin);
            else
                add_mutat(*begin - graph.numberOfVertices());
        }
    }

    void reset()
    {
        std::fill(mutat_edges.begin(), mutat_edges.end(), 0);
        std::fill(labels.begin(), labels.end(), 1);
        num_comp = 1;
    }

    const std::vector<size_t>& vertex_labels() const
    {
        return labels;
    }

    std::vector<size_t> separator() const
    {
        std::vector<size_t> separator(labels.size(), 0);
        for (size_t i = 0; i < labels.size(); ++i)
            if (labels[i] == 0)
                separator[i] = 1;
        return separator;
    }



private:
    
    const GRAPH& graph;
    const INTERACTION_GRAPH& interaction_graph;
    std::vector<size_t> mutat_edges;
    std::vector<size_t> labels;
    size_t num_comp;

    // This method labels the edge e as a mutual attraction edge if both end-vertices of the edge have the same label
    void add_mutat(size_t e)
    {
        size_t u = interaction_graph.vertexOfEdge(e, 0);
        if (labels[u] == 0)
            return;
        size_t v = interaction_graph.vertexOfEdge(e, 1);
        if (labels[u] == labels[v])
            mutat_edges[e] = 1;

        // TODO: Utilize a union find data structure that keeps track of the mutat components:
        //  E.g.: If there are two mutat constraints ab and bc then the mutat constraint ac does not 
        //  need to be added as it is already implied by ab and bc.
    }

    // This method adds vertex v to the separator if this does not cause a mutat edge to be separated
    void add_vertex_to_separator(size_t s)
    {
        // check if the vertex s has an outgoing mutat edge, if yes the vertex cannot be added to the separator
        for (auto it = interaction_graph.edgesFromVertexBegin(s); it != interaction_graph.edgesFromVertexEnd(s); ++it)
            if (mutat_edges[*it] == 1)
                return;

        // get all neighbors of s that are not in the separator
        std::vector<size_t> neighbors;
        for (auto it = graph.verticesFromVertexBegin(s); it != graph.verticesFromVertexEnd(s); ++it)
            if (labels[*it] > 0)
                neighbors.push_back(*it);

        // if there is at most one neighboring vertex, s can be added to the separator
        if (neighbors.size() <= 1) 
        {
            labels[s] = 0;
            return;
        }

        // add s to the separator (and potentially remove it again later)
        size_t label_s = labels[s];
        labels[s] = 0;

        // setup a bfs from all neighbors to check if s is a separating vertex
        std::vector<std::queue<size_t>> queues(neighbors.size());  // one queue for each bfs
        std::vector<size_t> new_labels(neighbors.size());  // label to mark in the labels array which vertices have been visited by each bfs
        for (size_t i = 0; i < neighbors.size(); ++i)
        {
            new_labels[i] = std::numeric_limits<size_t>::max() - i;
            labels[neighbors[i]] = new_labels[i];
            queues[i].push(neighbors[i]);
        }

        auto check_mutat = [this] (size_t vertex, size_t new_label)
        {
            auto e_it = interaction_graph.edgesFromVertexBegin(vertex);
            for (auto v_it = interaction_graph.verticesFromVertexBegin(vertex); v_it < interaction_graph.verticesFromVertexEnd(vertex); ++v_it, ++e_it)
            {
                if (mutat_edges[*e_it] == 0)
                    continue;
                if ((labels[*v_it] == labels[vertex]) || (labels[*v_it] == new_label))
                    continue;
                return true;
            }
            return false;
        };

        auto relabel = [this, &check_mutat] (size_t start, size_t new_label, bool check = false)
        {
            if (check && check_mutat(start, new_label))
                return true;

            // bfs from start to set all labels to new_label
            if (labels[start] == new_label)
                return false;
            size_t old_label = labels[start];   
            std::queue<size_t> queue;
            queue.push(start);
            labels[start] = new_label;

            while (queue.size() > 0)
            {
                size_t w = queue.front();
                queue.pop();
                // iterate over all neighbors of v and add them to the queue
                for(auto it = graph.verticesFromVertexBegin(w); it != graph.verticesFromVertexEnd(w); ++it)
                {
                    if (labels[*it] == old_label)
                    {
                        if (check && check_mutat(*it, new_label))
                            return true;

                        labels[*it] = new_label;
                        queue.push(*it);
                    }
                }
            }
            return false;
        };

        size_t empty_queues = 0;  // count the number of empty queues
        // start bfs
        while (empty_queues < neighbors.size() - 1)
        {
            for (size_t i = 0; i < neighbors.size(); ++i)
            {
                if (queues[i].size() == 0)
                    continue;
                
                size_t v = queues[i].front();
                queues[i].pop();
                // iterate over all neighbors of v and add them to the queue
                for(auto it = graph.verticesFromVertexBegin(v); it != graph.verticesFromVertexEnd(v); ++it)
                {
                    // if *it is in the separator, do nothing
                    if (labels[*it] == 0)
                        continue;
                    // if *it was already visited by the current bfs, do nothing
                    else if (labels[*it] == new_labels[i])
                        continue;
                    // if *it is still labeled with label_s, then proceed with normal bfs
                    else if (labels[*it] == label_s)
                    {
                        queues[i].push(*it);
                        labels[*it] = new_labels[i];
                        
                    }
                    // otherwise, the label of *it is equal to another new_label. This means that
                    // another bfs was found. The found bfs is added to the current bfs and then the
                    // respective queue is cleared.
                    else
                    {
                        // get the index j of the found bfs
                        size_t j = 0;
                        for (; j < neighbors.size(); ++j)
                            if (new_labels[j] == labels[*it])
                                break;
                        
                        // relabel all vertices labels new_labels[j] to new_labels[i]
                        // TODO: It would be better if this was also done in the interleaved fashion.
                        //  With the current approach the component corresponding the current bfs 
                        //  can become the size of (deg(g) - 1)/deg(g) of the component of label_s.
                        // NOTE: With the current implementation the labels are not consecutive, anyways because
                        //  it is possible that all vertices from one component are being added to the separator.
                        relabel(*it, new_labels[i], false);
                        // clear queue j into queue i
                        while (queues[j].size() > 0)
                        {
                            queues[i].push(queues[j].front());
                            queues[j].pop();
                        }
                        ++empty_queues;
                    }
                }

                // check if the current bfs just finished
                if (queues[i].size() == 0)
                {
                    // check for violated mutat constraints
                    if(relabel(neighbors[i], ++num_comp, true))
                    {   
                        relabel(neighbors[i], new_labels[i], false); // undo the re-labeling
                        goto FOUND_MUTAT_VIOLATION;
                    }

                    // increase the empty queues counter
                    ++empty_queues;
                }

                if (empty_queues >= neighbors.size() - 1)
                    break;
            }
        }
        

        // For the non-empty queue: undo the corresponding bfs by 
        // changing all labels back to the original label label_s
        for (size_t i = 0; i < neighbors.size(); ++i)
            if (queues[i].size() > 0)
                relabel(neighbors[i], label_s, false);
        return;

        FOUND_MUTAT_VIOLATION:
        {
            labels[s] = label_s;
            for (size_t i = 0; i < neighbors.size(); ++i)
                relabel(neighbors[i], label_s, false);
        }
    }
};


}
