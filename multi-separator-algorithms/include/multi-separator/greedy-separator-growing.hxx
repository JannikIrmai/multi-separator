#include <vector>
#include <array>
#include <queue>
#include <limits>
#include <map>
#include <set>
#include <iostream>

/*
Greedy separator growing algorithm: 
- all long range interactions (i.e. interactions that are not edges in the base graph) are attractive (i.e. positive costs).
    The costs of all other variables can be positive or negative.

- algorithm idea: Start with all vertices not in the separator, i.e. one large component. In each iteration, add the vertex 
    to the separator that yields the larges possible improvement in the objective value if such a vertex exists.
    1. Key idea: Instead of recomputing the improvement of each vertex every time one vertex is added to the separator we
    only update the improvements of those vertices which are affected.
    2. Key idea: Since all long range interactions are attractive they only make the improvement worse. These long range
    interaction costs are ignored when updating improvements and only before adding a vertex to the separator it is checked
    whether that vertex is a cut vertex and if so the long range interaction costs are considered.

- Pseudo code of algorithm:
    Compute potential of each vertex in initial solution

    While there exists a vertex with positive potential:
        Select the vertex v with the highest potential
        If v is a cut vertex:
            Subtract the cost of all long range interactions that are separated by v from the potential of v
            If v is no longer the vertex with the highest potential:
                continue
            recompute the potentials of all vertices that are in the same components as v was
            add v to the separator
        else:
            Add v to the separator
            Update the potentials of all vertices that are adjacent to v in the interaction graph and are in the same component as v
        
- The bottlenecks of the algorithm:
    - checking in each iteration whether v is a cut-vertex (can be parallelized by using on thread for each neighbor of v)
*/

namespace multi_separator
{


template<typename GRAPH, typename INTERACTION_GRAPH, typename T = double>
class GreedySeparatorGrowing
{
public:
    typedef GRAPH Graph;
    typedef INTERACTION_GRAPH InteractionGraph;
    typedef T value_type;

    struct Vertex
    {
        Vertex(size_t _v, value_type _p)
        {
            v = _v;
            p = _p;
            edition = 0;
        }
        size_t v;  // vertex index
        size_t edition;
        value_type p;  // vertex potential

        bool operator <(Vertex const& other) const
        {
            return p < other.p;
        }
    };

    GreedySeparatorGrowing(
        const Graph& graph, 
        const InteractionGraph& interaction_graph, 
        const std::vector<value_type> &vertex_costs,
        const std::vector<value_type> &interaction_costs
    ):
        graph(graph),
        interaction_graph(interaction_graph),
        vertex_costs(vertex_costs),
        interaction_costs(interaction_costs),
        labels(graph.numberOfVertices(), 1),
        temp_labels(graph.numberOfVertices(), 0),
        potentials(graph.numberOfVertices(), value_type()),
        vertex_editions(graph.numberOfVertices()),
        vertex2interactions(graph.numberOfVertices()),
        interaction2vertices(interaction_graph.numberOfEdges()),
        num_comp(1),
        num_separator(0),
        objective(value_type())
    {
        if (graph.numberOfVertices() != interaction_graph.numberOfVertices())
            throw std::runtime_error("Number of vertices of graph and interaction graph not equal.");
        if (graph.numberOfVertices() != vertex_costs.size())
            throw std::runtime_error("Number of vertices of graph and size of interaction costs not equal");
        if (interaction_graph.numberOfEdges() != interaction_costs.size())
            throw std::runtime_error("Number of edges of interaction graph and size of interaction costs not equal.");
        recompute_potentials();
    }

    void reset()
    {
        std::fill(labels.begin(), labels.end(), 1);
        num_comp = 1;
        num_separator = 0;
        objective = value_type();
        recompute_potentials();
    }

    void set_separator(const std::vector<size_t> separator)
    {
        throw std::runtime_error("Setting separator not yet implemented");
    }

    const std::vector<size_t>& vertex_labels() const
    {
        return labels;
    }

    size_t num_iter() const
    {
        return num_separator;
    }

    std::vector<size_t> separator() const
    {
        std::vector<size_t> separator(labels.size(), 0);
        for (size_t i = 0; i < labels.size(); ++i)
            if (labels[i] == 0)
                separator[i] = 1;
        return separator;
    }

        
    void run(size_t max_num_iter = -1)
    {
        size_t num_iter_start = num_separator;
        while (queue.size() > 0 && num_separator < num_iter_start + max_num_iter)
        {
            Vertex vertex = queue.top();
            queue.pop();
            if (vertex.edition != vertex_editions[vertex.v])
                continue;
            if (vertex.p < value_type()) // TODO: This should not be necessary if all nodes in queue have pos potential
                break;
            bool added = add_vertex_to_separator(vertex.v);
            if (added)
            {
                ++num_separator;
            }
            // printout the progress as a percentage
            if (graph.numberOfVertices() >= 100 && num_separator % (graph.numberOfVertices() / 100) == 0){
                auto percent = (num_separator * 100) / graph.numberOfVertices();
                std::cout << "\r" << percent << " %" << std::flush;
            }
        }
        std::cout << "\r";  // clear the percentage print out
    }

    value_type get_objective() const
    {
        return objective;
    }

    void remove_small_components(size_t threshold)
    {
        if (threshold == 0) 
            return;

        for (size_t u = 0; u < graph.numberOfVertices(); ++u)
        {
            if (labels[u] == 0 || temp_labels[u] != 0)
                continue;
            size_t comp_size = 1;
            std::queue<size_t> queue;
            queue.push(u);
            temp_labels[u] = 1;
            while (queue.size() > 0)
            {   
                size_t v = queue.front();
                queue.pop();
                for (auto w = graph.verticesFromVertexBegin(v); w < graph.verticesFromVertexEnd(v); ++w)
                {
                    if (labels[*w] == 0 || temp_labels[*w] != 0)
                        continue;
                    temp_labels[*w] = 1;
                    comp_size += 1;
                    queue.push(*w);
                }
            }
            if (comp_size > threshold)
                continue;
            
            queue.push(u);
            labels[u] = 0;
            while (queue.size() > 0)
            {   
                size_t v = queue.front();
                queue.pop();
                ++vertex_editions[v];
                for (auto w = graph.verticesFromVertexBegin(v); w < graph.verticesFromVertexEnd(v); ++w)
                {
                    if (labels[*w] == 0)
                        continue;
                    labels[*w] = 0;
                    queue.push(*w);
                }
            }
        }
        std::fill(temp_labels.begin(), temp_labels.end(), 0);
    }

    value_type get_potential(size_t v) const
    {
        return potentials[v];
    }

    void recompute_potentials()
    {
        // reset the queue
        queue = std::priority_queue<Vertex>();
        std::fill(vertex_editions.begin(), vertex_editions.end(), 0);
        // reset the interaction2vertices vector
        for (auto& vertices : interaction2vertices)
            vertices = std::set<size_t>();
        // iterate over all vertices, compute its potential and add it to the queue with a new edition
        for (size_t v = 0; v < graph.numberOfVertices(); ++v)
        {
            vertex2interactions[v] = std::set<size_t>();
            value_type potential;
            // if the vertex is already in the separator it cannot be added to the separator again and
            // thus has a potential of -infinity.
            if (labels[v] == 0)
                potential = -std::numeric_limits<value_type>::infinity();
            else
            {
                // otherwise the potential is computed as the minus the costs of v minus the sum of the costs of all interactions
                // between v and a neighbor w with the same label. This is because currently the interaction vw is not separated
                // and adding v to the separator does separate the interaction.
                potential = -vertex_costs[v];
                for (auto adj = interaction_graph.adjacenciesFromVertexBegin(v); adj < interaction_graph.adjacenciesFromVertexEnd(v); ++adj)
                {
                    if (labels[adj->vertex()] == labels[v])
                    {
                        potential -= interaction_costs[adj->edge()];
                        vertex2interactions[v].insert(adj->edge());
                        interaction2vertices[adj->edge()].insert(v);
                    }
                }
            }
            potentials[v] = potential;
            if (potential >= value_type())
                queue.push(Vertex(v, potential));
        }
    }

private:
    
    Graph graph;
    InteractionGraph interaction_graph;
    std::vector<value_type> vertex_costs;
    std::vector<value_type> interaction_costs;
    std::vector<size_t> labels;
    std::vector<size_t> temp_labels;
    std::vector<value_type> potentials;
    size_t num_comp;
    size_t num_separator;
    value_type objective;

    std::priority_queue<Vertex> queue;
    std::vector<size_t> vertex_editions;

    // this vector stores for each vertex the set of those interactions
    // for which the vertex is a cut-vertex.
    std::vector<std::set<size_t>> vertex2interactions;
    // this vector stores for each interaction the set of all vertices which
    // are cut vertices of the given interaction.
    std::vector<std::set<size_t>> interaction2vertices;

    // This method adds vertex v to the separator if this does not cause a mutat edge to be separated
    bool add_vertex_to_separator(const size_t s)
    {
        if (labels[s] == 0)
            throw std::runtime_error("Vertex is already in separator");
        // get all neighbors of s that are not in the separator
        std::vector<size_t> neighbors;
        for (auto it = graph.verticesFromVertexBegin(s); it != graph.verticesFromVertexEnd(s); ++it)
            if (labels[*it] > 0)
                neighbors.push_back(*it);

        const size_t label_s = labels[s];
        size_t old_num_comp = num_comp;

        auto update_neighboring_potentials = [this, &s] ()
        {
            // For each interaction for which s is a cut vertex, update the potentials of all vertices
            // that are also cut-vertices of that interaction
            for (size_t e : vertex2interactions[s])
            {
                for (size_t v : interaction2vertices[e])
                {
                    if (v == s)
                        continue;
                    potentials[v] += interaction_costs[e];
                    Vertex vertex(v, potentials[v]);
                    vertex.edition = ++vertex_editions[v];
                    vertex2interactions[v].erase(e);
                    if (potentials[v] >= value_type())
                        queue.push(vertex);

                }
                interaction2vertices[e] = std::set<size_t>();
            }
            vertex2interactions[s] = std::set<size_t>();
        };

        // add s to the separator (and potentially remove it again later)
        labels[s] = 0;

        // if there is at most one neighboring vertex, s can be added to the separator
        if (neighbors.size() <= 1) 
        {
            update_neighboring_potentials();
            objective -= potentials[s];
            return true;
        }

        value_type potential = potentials[s];

        // setup a bfs from all neighbors to check if s is a separating vertex
        std::vector<std::queue<size_t>> queues(neighbors.size());  // one queue for each bfs
        std::vector<size_t> new_labels(neighbors.size());  // labels to mark in the temp_labels array which vertices have been visited by each bfs
        for (size_t i = 0; i < neighbors.size(); ++i)
        {
            new_labels[i] = i+1;
            temp_labels[neighbors[i]] = new_labels[i];
            queues[i].push(neighbors[i]);
        }

        // this method updates the potential of s by the cost of all interactions from v to n where n has a 
        // different label than v in temp_labels and labels[n] == label_s. These are all interaction that
        // are adjacent to v that become separated by adding s to the separator.
        auto update_potential = [this, &s, &label_s, &potential] (size_t v)
        {
            for (auto adj = interaction_graph.adjacenciesFromVertexBegin(v); adj < interaction_graph.adjacenciesFromVertexEnd(v); ++adj)
                if ((temp_labels[v] != temp_labels[adj->vertex()]) && (labels[adj->vertex()] == label_s))
                {
                    size_t e = adj->edge();
                    if (interaction2vertices[e].count(s) == 0)
                    {   
                        interaction2vertices[e].insert(s);
                        vertex2interactions[s].insert(e);
                        potential -= interaction_costs[e];
                    }
                }
        };

        // this method relabels by bfs
        auto relabel = [this, &update_potential] (std::vector<size_t>& labels_, size_t start, size_t new_label, bool update_potential_flag)
        {
            // bfs from start to set all labels to new_label
            if (labels_[start] == new_label)
                return;

            if (update_potential_flag)
                update_potential(start);

            size_t old_label = labels_[start];   
            std::queue<size_t> queue;
            queue.push(start);
            labels_[start] = new_label;

            while (queue.size() > 0)
            {
                size_t w = queue.front();
                queue.pop();
                // iterate over all neighbors of v and add them to the queue
                for(auto it = graph.verticesFromVertexBegin(w); it != graph.verticesFromVertexEnd(w); ++it)
                {
                    if (labels_[*it] == old_label)
                    {
                        if (update_potential_flag)
                            update_potential(*it);
                        labels_[*it] = new_label;
                        queue.push(*it);
                    }
                }
            }
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
                    else if (temp_labels[*it] == new_labels[i])
                        continue;
                    // if *it is still labeled with 0 in temp_labels, then proceed with normal bfs
                    else if (temp_labels[*it] == 0)
                    {
                        queues[i].push(*it);
                        temp_labels[*it] = new_labels[i];        
                    }
                    // otherwise, the label of *it is equal to another new_label. This means that
                    // another bfs was found. The found bfs is added to the current bfs and then the
                    // respective queue is cleared.
                    else
                    {
                        // get the index j of the found bfs
                        size_t j = 0;
                        for (; j < neighbors.size(); ++j)
                            if (new_labels[j] == temp_labels[*it])
                                break;
                        // relabel all vertices labeled new_labels[j] to new_labels[i]
                        // TODO: It would be better if this was also done in the interleaved fashion.
                        //  With the current approach the component corresponding the current bfs 
                        //  can become the size of (deg(g) - 1)/deg(g) of the component of label_s.
                        // NOTE: With the current implementation the labels are not consecutive, anyways because
                        //  it is possible that all vertices from one component are being added to the separator.
                        relabel(temp_labels, *it, new_labels[i], false);
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
                    // update the actual labels and update the potential of s
                    relabel(labels, neighbors[i], ++num_comp, true);

                    // increase the empty queues counter
                    ++empty_queues;
                }
                if (empty_queues >= neighbors.size() - 1)
                    break;
            }
        }

        // get the next best potential in the queue
        value_type next_best_potential = 0;
        while (queue.size() > 0)
        {
            Vertex vertex = queue.top();
            if (vertex.edition != vertex_editions[vertex.v])
            {
                queue.pop();
                continue;
            }
            next_best_potential = vertex.p;
            break;
        }

        // check if the potential of s is now less than next_best_potential
        if (potential >= next_best_potential)
        {    
            for (size_t i = 0; i < neighbors.size(); ++i)
            {
                relabel(temp_labels, neighbors[i], 0, false);
                if (queues[i].size() > 0)
                {
                    relabel(labels, neighbors[i], label_s, false);
                }
            }

            update_neighboring_potentials();
            objective -= potential;   
            return true;
        }
        else
        {
            labels[s] = label_s;
            for (size_t i = 0; i < neighbors.size(); ++i)
            {
                relabel(temp_labels, neighbors[i], 0, false);
                relabel(labels, neighbors[i], label_s, false);
            }
            num_comp = old_num_comp;
            potentials[s] = potential;
            Vertex vertex(s, potential);
            vertex.edition = ++vertex_editions[s];
            if (potential >= value_type())
                queue.push(vertex);
            return false;
        }
    }
};

} // multi_separator