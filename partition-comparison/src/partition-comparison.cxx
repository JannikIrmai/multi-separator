#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <andres/partition-comparison.hxx>

namespace py = pybind11;

typedef andres::RandError<> RandError;
typedef andres::PartialRandError<> PartialRandError;
typedef andres::VariationOfInformation<> VariationOfInformation;
typedef andres::PartialVariationOfInformation<> PartialVariationOfInformation;
typedef andres::SingletonVariationOfInformation<> SingletonVariationOfInformation;
typedef andres::WeightedSingletonVariationOfInformation<> WeightedSingletonVariationOfInformation;


// custom constructors for easier usage in python
// TODO: it is unnecessary to specify size_t?

RandError rand_error
(
    const py::array_t<size_t>& labels0,
    const py::array_t<size_t>& labels1,
    bool ignoreDefaultLabel = true
) 
{
    if (labels0.size() != labels1.size())
        throw std::runtime_error("size of inputs not equal.");

    return RandError(labels0.data(), labels0.data() + labels0.size(), labels1.data(), ignoreDefaultLabel);
}

PartialRandError partial_rand_error
(
    const py::array_t<size_t>& coarse_labels,
    const py::array_t<size_t>& fine_labels,
    const py::array_t<size_t>& pred_labels,
    bool ignoreDefaultLabel = true
) 
{
    if (coarse_labels.size() != fine_labels.size() || coarse_labels.size() != pred_labels.size())
        throw std::runtime_error("size of inputs not equal.");

    return PartialRandError(coarse_labels.data(), coarse_labels.data() + coarse_labels.size(), fine_labels.data(), pred_labels.data(), ignoreDefaultLabel);
}

VariationOfInformation variation_of_information
(
    const py::array_t<size_t>& labels0,
    const py::array_t<size_t>& labels1,
    bool ignoreDefaultLabel = true
) 
{
    if (labels0.size() != labels1.size())
        throw std::runtime_error("size of inputs not equal.");

    return VariationOfInformation(labels0.data(), labels0.data() + labels0.size(), labels1.data(), ignoreDefaultLabel);
}

PartialVariationOfInformation partial_variation_of_information
(
    const py::array_t<size_t>& coarse_labels,
    const py::array_t<size_t>& fine_labels,
    const py::array_t<size_t>& pred_labels,
    bool ignoreDefaultLabel = true
) 
{
    if (coarse_labels.size() != fine_labels.size() || coarse_labels.size() != pred_labels.size())
        throw std::runtime_error("size of inputs not equal.");

    return PartialVariationOfInformation(coarse_labels.data(), coarse_labels.data() + coarse_labels.size(), fine_labels.data(), pred_labels.data(), ignoreDefaultLabel);
}

SingletonVariationOfInformation singleton_variation_of_information
(
    const py::array_t<size_t>& truth,
    const py::array_t<size_t>& pred
) 
{
    if (truth.size() != pred.size())
        throw std::runtime_error("size of inputs not equal.");

    return SingletonVariationOfInformation(truth.data(), truth.data() + truth.size(), pred.data());
}

WeightedSingletonVariationOfInformation weighted_singleton_variation_of_information
(
    const py::array_t<size_t>& truth,
    const py::array_t<size_t>& pred,
    const py::array_t<double>& weight
) 
{
    if (truth.size() != pred.size() || weight.size() != truth.size())
        throw std::runtime_error("size of inputs not equal.");

    return WeightedSingletonVariationOfInformation(truth.data(), truth.data() + truth.size(), pred.data(), weight.data());
}


// wrapping cpp classes to use in python
PYBIND11_MODULE(partition_comparison, m)
{
    m.doc() = "pybind11 partition_comparison module";

    py::class_<RandError>(m, "RandError")
        .def(py::init(&rand_error), "Constructor", py::arg("labels0"), py::arg("labels1"), py::arg("ignoreDefaultLabel") = true)
        .def("elements", &RandError::elements)
        .def("pairs", &RandError::pairs)
        .def("trueJoins", &RandError::trueJoins)
        .def("trueCuts", &RandError::trueCuts)
        .def("falseJoins", &RandError::falseJoins)
        .def("falseCuts", &RandError::falseCuts)
        .def("joinsInPrediction", &RandError::joinsInPrediction)
        .def("cutsInPrediction", &RandError::cutsInPrediction)
        .def("joinsInTruth", &RandError::joinsInTruth)
        .def("cutsInTruth", &RandError::cutsInTruth)
        .def("recallOfCuts", &RandError::recallOfCuts)
        .def("precisionOfCuts", &RandError::precisionOfCuts)
        .def("recallOfJoins", &RandError::recallOfJoins)
        .def("precisionOfJoins", &RandError::precisionOfJoins)
        .def("error", &RandError::error)
        .def("index", &RandError::index);

    py::class_<PartialRandError>(m, "PartialRandError")
        .def(py::init(&partial_rand_error), "Constructor", py::arg("coarse"), py::arg("fine"), py::arg("pred"), py::arg("ignoreDefaultLabel") = true)
        .def("elements", &PartialRandError::elements)
        .def("pairs", &PartialRandError::pairs)
        .def("trueJoins", &PartialRandError::trueJoins)
        .def("trueCuts", &PartialRandError::trueCuts)
        .def("falseJoins", &PartialRandError::falseJoins)
        .def("falseCuts", &PartialRandError::falseCuts)
        .def("joinsInPrediction", &PartialRandError::joinsInPrediction)
        .def("cutsInPrediction", &PartialRandError::cutsInPrediction)
        .def("joinsInTruth", &PartialRandError::joinsInTruth)
        .def("cutsInTruth", &PartialRandError::cutsInTruth)
        .def("recallOfCuts", &PartialRandError::recallOfCuts)
        .def("precisionOfCuts", &PartialRandError::precisionOfCuts)
        .def("recallOfJoins", &PartialRandError::recallOfJoins)
        .def("precisionOfJoins", &PartialRandError::precisionOfJoins)
        .def("error", &PartialRandError::error)
        .def("index", &PartialRandError::index);

    py::class_<VariationOfInformation>(m, "VariationOfInformation")
        .def(py::init(&variation_of_information), "Constructor", py::arg("labels0"), py::arg("labels1"), py::arg("ignoreDefaultLabel") = true)
        .def("value", &VariationOfInformation::value)
        .def("valueFalseCut", &VariationOfInformation::valueFalseCut)
        .def("valueFalseJoin", &VariationOfInformation::valueFalseJoin);

    py::class_<PartialVariationOfInformation>(m, "PartialVariationOfInformation")
        .def(py::init(&partial_variation_of_information), "Constructor", py::arg("coarse"), py::arg("fine"), py::arg("pred"), py::arg("ignoreDefaultLabel") = true)
        .def("value", &PartialVariationOfInformation::value)
        .def("valueFalseCut", &PartialVariationOfInformation::valueFalseCut)
        .def("valueFalseJoin", &PartialVariationOfInformation::valueFalseJoin);

    py::class_<SingletonVariationOfInformation>(m, "SingletonVariationOfInformation")
        .def(py::init(&singleton_variation_of_information), "Constructor", py::arg("truth"), py::arg("pred"))
        .def("value", &SingletonVariationOfInformation::value)
        .def("valueFalseCut", &SingletonVariationOfInformation::valueFalseCut)
        .def("valueFalseJoin", &SingletonVariationOfInformation::valueFalseJoin);

    py::class_<WeightedSingletonVariationOfInformation>(m, "WeightedSingletonVariationOfInformation") 
        .def(py::init(&weighted_singleton_variation_of_information), "Constructor", py::arg("truth"), py::arg("pred"), py::arg("weight"))
        .def("value", &WeightedSingletonVariationOfInformation::value)
        .def("valueFalseCut", &WeightedSingletonVariationOfInformation::valueFalseCut)
        .def("valueFalseJoin", &WeightedSingletonVariationOfInformation::valueFalseJoin);
}