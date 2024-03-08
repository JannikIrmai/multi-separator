#pragma once
#ifndef ANDRES_PARTITION_COMPARISON_HXX
#define ANDRES_PARTITION_COMPARISON_HXX

#include <map>
#include <utility> // pair
#include <iterator> // iterator_traits
#include <cmath> // log
#include <stdexcept> // runtime_error


namespace andres {

template<class T = double>
class RandError {
public:
    typedef T value_type;

    template<class ITERATOR_TRUTH, class ITERATOR_PRED>
    RandError(ITERATOR_TRUTH begin0, ITERATOR_TRUTH end0, ITERATOR_PRED begin1, bool ignoreDefaultLabel = false)
    {
        typedef typename std::iterator_traits<ITERATOR_TRUTH>::value_type Label0;
        typedef typename std::iterator_traits<ITERATOR_PRED>::value_type Label1;
        typedef std::pair<Label0, Label1> Pair;
        typedef std::map<Pair, size_t> OverlapMatrix;
        typedef std::map<Label0, size_t> TruthSumMap;
        typedef std::map<Label1, size_t> PredSumMap;

        OverlapMatrix n;
        TruthSumMap truthSum;
        PredSumMap predSum;

        elements_ = std::distance(begin0, end0);

        if (ignoreDefaultLabel)
        {
            elements_ = 0;

            for(; begin0 != end0; ++begin0, ++begin1)
                if (*begin0 != Label0() && *begin1 != Label1())
                {
                    ++n[Pair(*begin0, *begin1)];
                    ++truthSum[*begin0];
                    ++predSum[*begin1];
                    ++elements_;
                }

            if (elements_ == 0)
                throw std::runtime_error("No element is labeled in both partitions.");
        }
        else
            for(; begin0 != end0; ++begin0, ++begin1)
            {
                ++n[Pair(*begin0, *begin1)];
                ++truthSum[*begin0];
                ++predSum[*begin1];
            }

        for (auto const& it : predSum)
            falseJoins_ += it.second * it.second;

        for (auto const& it : truthSum)
            falseCuts_ += it.second * it.second;

        for (auto const& it : n)
        {
            const size_t n_ij = it.second;

            trueJoins_ += n_ij * (n_ij - 1) / 2;
            falseCuts_ -= n_ij * n_ij;
            falseJoins_ -= n_ij * n_ij;
        }

        falseJoins_ /= 2;
        falseCuts_ /= 2;

        trueCuts_ = pairs() - joinsInPrediction() - falseCuts_;
    }

    size_t elements() const
        { return elements_; }
    size_t pairs() const
        { return elements_ * (elements_ - 1) / 2; }

    size_t trueJoins() const
        { return trueJoins_; }
    size_t trueCuts() const
        { return trueCuts_; }
    size_t falseJoins() const
        { return falseJoins_; }
    size_t falseCuts() const
        { return falseCuts_; }

    size_t joinsInPrediction() const
        { return trueJoins_ + falseJoins_; }
    size_t cutsInPrediction() const
        { return trueCuts_ + falseCuts_; }
    size_t joinsInTruth() const
        { return trueJoins_ + falseCuts_; }
    size_t cutsInTruth() const
        { return trueCuts_ + falseJoins_; }

    value_type recallOfCuts() const
        {
            if(cutsInTruth() == 0)
                return 1;
            else
                return static_cast<value_type>(trueCuts()) / cutsInTruth();
        }
    value_type precisionOfCuts() const
        {
            if(cutsInPrediction() == 0)
                return 1;
            else
                return static_cast<value_type>(trueCuts()) / cutsInPrediction();
        }

    value_type recallOfJoins() const
        {
            if(joinsInTruth() == 0)
                return 1;
            else
                return static_cast<value_type>(trueJoins()) / joinsInTruth();
        }
    value_type precisionOfJoins() const
        {
            if(joinsInPrediction() == 0)
                return 1;
            else
                return static_cast<value_type>(trueJoins()) / joinsInPrediction();
        }

    value_type error() const
        { return static_cast<value_type>(falseJoins() + falseCuts()) / pairs(); }
    value_type index() const
        { return static_cast<value_type>(trueJoins() + trueCuts()) / pairs(); }

private:
    size_t elements_;
    size_t trueJoins_ { size_t() };
    size_t trueCuts_ { size_t() };
    size_t falseJoins_ { size_t() };
    size_t falseCuts_ { size_t() };
};



template<class T = double>
class PartialRandError {
public:

    template<class ITERATOR_TRUTH, class ITERATOR_PRED>
    PartialRandError(ITERATOR_TRUTH begin_coarse, ITERATOR_TRUTH end, ITERATOR_TRUTH begin_fine, ITERATOR_PRED begin_pred, bool ignoreDefaultLabel = false)
    {
        // TODO: the default label has different meanings for the coarse and fine partition:
        // for the coarse partition 0 means ignore, for the fine partition 0 means singleton cluster.

        typedef typename std::iterator_traits<ITERATOR_TRUTH>::value_type LabelGT;
        typedef typename std::iterator_traits<ITERATOR_PRED>::value_type LabelPred;
        typedef std::pair<LabelGT, LabelPred> Pair;
        typedef std::map<Pair, size_t> OverlapMatrix;
        typedef std::map<LabelGT, size_t> TruthSumMap;
        typedef std::map<LabelPred, size_t> PredSumMap;

        OverlapMatrix coarse_matrix;
        OverlapMatrix fine_matrix;
        TruthSumMap coarse_sum;
        TruthSumMap fine_sum;
        PredSumMap pred_sum;

        std::map<LabelGT, LabelGT> super_set;

        elements_ = 0;

        for(; begin_coarse != end; ++begin_coarse, ++begin_fine, ++begin_pred)
        {
            if (ignoreDefaultLabel && (*begin_coarse == LabelGT() || *begin_pred == LabelPred()))
                continue;
            
            ++coarse_matrix[Pair(*begin_coarse, *begin_pred)];
            ++coarse_sum[*begin_coarse];
            ++pred_sum[*begin_pred];
            ++elements_;

            if (ignoreDefaultLabel && *begin_fine == LabelGT())
                continue;

            ++fine_sum[*begin_fine];
            ++fine_matrix[Pair(*begin_fine, *begin_pred)];
            
            if (super_set.count(*begin_fine) == 0)
                super_set[*begin_fine] = *begin_coarse;
            else if (super_set[*begin_fine] != *begin_coarse)
                throw std::runtime_error("Ground truth is not a coarse-fine relationship");
        }
        
        if (elements_ == 0)
            throw std::runtime_error("No element is labeled in both partitions.");

        for (auto const& it : pred_sum)
            falseJoins_ += it.second * it.second;

        for (auto const& it : fine_sum)
            falseCuts_ += it.second * it.second;

        for (auto const& it : coarse_matrix)
        {
            const size_t n_ij = it.second;
            trueJoins_ += n_ij * (n_ij - 1) / 2;
            falseJoins_ -= n_ij * n_ij;
        }

        for (auto const& it : fine_matrix)
        {
            const size_t n_ij = it.second;
            falseCuts_ -= n_ij * n_ij;
        }

        falseJoins_ /= 2;
        falseCuts_ /= 2;

        trueCuts_ = pairs() - joinsInPrediction() - falseCuts_;
    }

    size_t elements() const
        { return elements_; }
    size_t pairs() const
        { return elements_ * (elements_ - 1) / 2; }

    size_t trueJoins() const
        { return trueJoins_; }
    size_t trueCuts() const
        { return trueCuts_; }
    size_t falseJoins() const
        { return falseJoins_; }
    size_t falseCuts() const
        { return falseCuts_; }

    size_t joinsInPrediction() const
        { return trueJoins_ + falseJoins_; }
    size_t cutsInPrediction() const
        { return trueCuts_ + falseCuts_; }
    size_t joinsInTruth() const
        { return trueJoins_ + falseCuts_; }
    size_t cutsInTruth() const
        { return trueCuts_ + falseJoins_; }

    T recallOfCuts() const
        {
            if(cutsInTruth() == 0)
                return 1;
            else
                return static_cast<T>(trueCuts()) / cutsInTruth();
        }
    T precisionOfCuts() const
        {
            if(cutsInPrediction() == 0)
                return 1;
            else
                return static_cast<T>(trueCuts()) / cutsInPrediction();
        }

    T recallOfJoins() const
        {
            if(joinsInTruth() == 0)
                return 1;
            else
                return static_cast<T>(trueJoins()) / joinsInTruth();
        }
    T precisionOfJoins() const
        {
            if(joinsInPrediction() == 0)
                return 1;
            else
                return static_cast<T>(trueJoins()) / joinsInPrediction();
        }

    T error() const
        { return static_cast<T>(falseJoins() + falseCuts()) / pairs(); }
    T index() const
        { return static_cast<T>(trueJoins() + trueCuts()) / pairs(); }

private:
    size_t elements_;
    size_t trueJoins_ { size_t() };
    size_t trueCuts_ { size_t() };
    size_t falseJoins_ { size_t() };
    size_t falseCuts_ { size_t() };
};
// TODO clean this up by collecting functionality of RandError and PartialRandError in super class


template<class T = double>
class VariationOfInformation {
public:
    typedef T value_type;

    template<class ITERATOR_TRUTH, class ITERATOR_PRED>
    VariationOfInformation(ITERATOR_TRUTH begin0, ITERATOR_TRUTH end0, ITERATOR_PRED begin1, bool ignoreDefaultLabel = false)
    {
        typedef typename std::iterator_traits<ITERATOR_TRUTH>::value_type Label0;
        typedef typename std::iterator_traits<ITERATOR_PRED>::value_type Label1;
        typedef std::pair<Label0, Label1> Pair;
        typedef std::map<Pair, double> PMatrix;
        typedef std::map<Label0, double> PVector0;
        typedef std::map<Label1, double> PVector1;

        // count
        size_t N = std::distance(begin0, end0);

        PMatrix pjk;
        PVector0 pj;
        PVector1 pk;

        if (ignoreDefaultLabel)
        {
            N = 0;

            for (; begin0 != end0; ++begin0, ++begin1)
                if (*begin0 != Label0() && *begin1 != Label1())
                {
                    ++pj[*begin0];
                    ++pk[*begin1];
                    ++pjk[Pair(*begin0, *begin1)];
                    ++N;
                }
        }
        else
            for (; begin0 != end0; ++begin0, ++begin1)
            {
                ++pj[*begin0];
                ++pk[*begin1];
                ++pjk[Pair(*begin0, *begin1)];
            }

        // normalize
        for (auto& p : pj)
            p.second /= N;
        
        for (auto& p : pk)
            p.second /= N;
        
        for (auto& p : pjk)
            p.second /= N;

        // compute information
        auto H0 = value_type();
        for (auto const& p : pj)
            H0 -= p.second * std::log2(p.second);
        
        auto H1 = value_type();
        for (auto const& p : pk)
            H1 -= p.second * std::log2(p.second);
        
        auto I = value_type();
        for (auto const& p : pjk)
        {
            auto j = p.first.first;
            auto k = p.first.second;
            auto pjk_here = p.second;
            auto pj_here = pj[j];
            auto pk_here = pk[k];

            I += pjk_here * std::log2( pjk_here / (pj_here * pk_here) );
        }

        value_ = H0 + H1 - 2.0 * I;
        precision_ = H1 - I;
        recall_ = H0 - I;
    }

    value_type value() const
    {
        return value_;
    }

    value_type valueFalseCut() const
    {
        return precision_;
    }

    value_type valueFalseJoin() const
    {
        return recall_;
    }

private:
    value_type value_{};
    value_type precision_{};
    value_type recall_{};
};


template<class T = double>
class PartialVariationOfInformation {
public:

    template<class ITERATOR_TRUTH, class ITERATOR_PRED>
    PartialVariationOfInformation(ITERATOR_TRUTH begin_coarse, ITERATOR_TRUTH end, ITERATOR_TRUTH begin_fine, ITERATOR_PRED begin_pred, bool ignoreDefaultLabel = false)
    {
        typedef typename std::iterator_traits<ITERATOR_TRUTH>::value_type LabelGT;
        typedef typename std::iterator_traits<ITERATOR_PRED>::value_type LabelPred;
        typedef std::pair<LabelGT, LabelPred> Pair;
        typedef std::map<Pair, double> OverlapMatrix;
        typedef std::map<LabelGT, double> TruthSumMap;
        typedef std::map<LabelPred, double> PredSumMap;


        OverlapMatrix p_coarse_pred;
        OverlapMatrix p_fine_pred;
        TruthSumMap p_fine;
        PredSumMap p_pred;

        std::map<LabelGT, LabelGT> super_set;

        size_t N = 0;

        for (; begin_coarse != end; ++begin_coarse, ++begin_fine, begin_pred)
        {
            if (ignoreDefaultLabel && (*begin_coarse == LabelGT() || *begin_pred == LabelPred()))
                continue;
            ++p_pred[*begin_pred];
            ++p_coarse_pred[Pair(*begin_coarse, *begin_pred)];
            ++N;
        
            if (ignoreDefaultLabel && *begin_fine == LabelGT())
                continue;

            ++p_fine[*begin_fine];
            ++p_fine_pred[Pair(*begin_fine, *begin_pred)];
            
            if (super_set.count(*begin_fine) == 0)
                super_set[*begin_fine] = *begin_coarse;
            else if (super_set[*begin_fine] != *begin_coarse)
                throw std::runtime_error("Ground truth is not a coarse-fine relationship");
        }
        
        // normalize
        for (auto& p : p_coarse_pred)
            p.second /= N;
        
        for (auto& p : p_fine_pred)
            p.second /= N;
        
        for (auto& p : p_fine)
            p.second /= N;

        for (auto& p : p_pred)
            p.second /= N;

        // compute conditional entropies
        for (auto const& p : p_coarse_pred)
            precision_ -= p.second * std::log2(p.second / p_pred[p.first.second]);

        T ce_pred_fine = T();
        for (auto const& p : p_fine_pred)
            recall_ -= p.second * std::log2(p.second / p_fine[p.first.first]);

        value_ = precision_ + recall_;
    }

    T value() const
    {
        return value_;
    }

    T valueFalseCut() const
    {
        return precision_;
    }

    T valueFalseJoin() const
    {
        return recall_;
    }

private:
    T value_{};
    T precision_{};
    T recall_{};

};


template<class T = double>
class SingletonVariationOfInformation {
public:

    template<class ITERATOR_TRUTH, class ITERATOR_PRED>
    SingletonVariationOfInformation(ITERATOR_TRUTH begin_truth, ITERATOR_TRUTH end, ITERATOR_PRED begin_pred)
    {
        typedef typename std::iterator_traits<ITERATOR_TRUTH>::value_type LabelTruth;
        typedef typename std::iterator_traits<ITERATOR_PRED>::value_type LabelPred;
        typedef std::pair<LabelTruth, LabelPred> Pair;
        typedef std::map<Pair, double> OverlapMatrix;
        typedef std::map<LabelTruth, double> TruthSumMap;
        typedef std::map<LabelPred, double> PredSumMap;


        OverlapMatrix p_truth_pred;
        TruthSumMap p_truth;
        PredSumMap p_pred;

        size_t N = std::distance(begin_truth, end);

        for (; begin_truth != end; ++begin_truth, ++begin_pred)
        {
            ++p_truth[*begin_truth];
            ++p_pred[*begin_pred];
            ++p_truth_pred[Pair(*begin_truth, *begin_pred)];
        }
        

        // compute conditional entropies (recall = H(truth|pred) and precision = H(pred|truth))
        for (auto const& p : p_truth)
        {
            if (p.first == LabelTruth())
                continue;
            precision_ -= p_truth_pred[Pair(p.first, LabelPred())] / N * std::log2(1/p.second);
        }

        for (auto const& p : p_pred)
        {
            if (p.first == LabelPred())
                continue;
            recall_ -= p_truth_pred[Pair(LabelTruth(), p.first)] / N * std::log2(1/p.second);
        }

        for (auto const& p : p_truth_pred)
        {
            if (p.first.first == LabelTruth() || p.first.second == LabelPred())
                continue;

            precision_ -= p.second / N * std::log2(p.second / p_truth[p.first.first]);
            recall_ -= p.second / N * std::log2(p.second / p_pred[p.first.second]);
        }

        value_ = precision_ + recall_;
    }

    T value() const
    {
        return value_;
    }

    T valueFalseCut() const
    {
        return precision_;
    }

    T valueFalseJoin() const
    {
        return recall_;
    }

private:
    T value_{};
    T precision_{};
    T recall_{};

};


template<class T = double>
class WeightedSingletonVariationOfInformation {
public:

    template<class ITERATOR_TRUTH, class ITERATOR_PRED, class ITERATOR_WEIGHT>
    WeightedSingletonVariationOfInformation(ITERATOR_TRUTH begin_truth, ITERATOR_TRUTH end, ITERATOR_PRED begin_pred, ITERATOR_WEIGHT weight_begin)
    {
        typedef typename std::iterator_traits<ITERATOR_TRUTH>::value_type LabelTruth;
        typedef typename std::iterator_traits<ITERATOR_PRED>::value_type LabelPred;
        typedef typename std::iterator_traits<ITERATOR_WEIGHT>::value_type WeightType;
        typedef std::pair<LabelTruth, LabelPred> Pair;
        typedef std::map<Pair, T> OverlapMatrix;
        typedef std::map<LabelTruth, T> TruthSumMap;
        typedef std::map<LabelPred, T> PredSumMap;

        /*
        Let X and Y be the truth and prediction.
        We compute the variation of information as VI(X, Y) = 2*H(X, Y) - H(X) - H(Y)
        Where H is the entropy.
        All elements that are labeled with the default label (i.e. 0) are assumed to be in singleton clusters.
        */

        OverlapMatrix p_joint;
        TruthSumMap p_truth;
        PredSumMap p_pred;

        T truth_entropy{};
        T pred_entropy{};
        T joint_entropy{};

        WeightType total_weight{};
        WeightType total_joint_singleton_weight{};
        WeightType total_truth_singleton_weight{};
        WeightType total_pred_singleton_weight{};

        // count the weighted number of occurrences of all non-singleton labels 
        // and compute the entropies with respect to the singleton labels
        for (; begin_truth != end; ++begin_truth, ++begin_pred, ++weight_begin)
        {
            if (*weight_begin == WeightType())
                continue;

            total_weight += *weight_begin;

            if (*begin_truth == LabelTruth())
            {
                truth_entropy -= *weight_begin * std::log2(*weight_begin);
                total_truth_singleton_weight += *weight_begin;
            }
            else
                p_truth[*begin_truth] += *weight_begin;

            if (*begin_pred == LabelPred())
            {
                pred_entropy -= *weight_begin * std::log2(*weight_begin);
                total_pred_singleton_weight += *weight_begin;
            }
            else
                p_pred[*begin_pred] += *weight_begin;


            if (*begin_truth == LabelTruth() || *begin_pred == LabelPred())
            {
                total_joint_singleton_weight += *weight_begin;
                joint_entropy -= *weight_begin * std::log2(*weight_begin);
            }
            else
                p_joint[Pair(*begin_truth, *begin_pred)] += *weight_begin;
        }        

        // normalize the entropies
        truth_entropy = truth_entropy / total_weight + std::log2(total_weight) / total_weight * total_truth_singleton_weight;
        pred_entropy = pred_entropy / total_weight + std::log2(total_weight) / total_weight * total_pred_singleton_weight;
        joint_entropy = joint_entropy / total_weight + std::log2(total_weight) / total_weight * total_joint_singleton_weight;

        // compute the entropies corresponding to the non-singleton labels
        for (auto const& p : p_truth)
            truth_entropy -= p.second / total_weight * std::log2(p.second/total_weight);

        for (auto const& p : p_pred)
            pred_entropy -= p.second / total_weight * std::log2(p.second/total_weight);

        for (auto const& p : p_joint)
            joint_entropy -= p.second / total_weight * std::log2(p.second/total_weight);

        
        false_cut_ = joint_entropy - truth_entropy;
        false_join_ = joint_entropy - pred_entropy;
    }

    T value() const
    {
        return false_cut_ + false_join_;
    }

    T valueFalseCut() const
    {
        return false_cut_;
    }

    T valueFalseJoin() const
    {
        return false_join_;
    }

private:
    T false_cut_{};
    T false_join_{};

};


} // namespace andres

#endif // #ifndef ANDRES_PARTITION_COMPARISON_HXX
