#ifndef C___RANDOM_UTILS_H
#define C___RANDOM_UTILS_H

#include <random>

namespace poweruct {

    class RandomUtils {

    public:
        RandomUtils();
        ~RandomUtils();
        void seed(uint32_t s);
        uint32_t sample_discrete(std::vector<double> &probabilities);
        double sample_uniform();
        uint32_t sample_uniform(uint32_t n);

        template<typename T>
        T sample_uniform(const std::vector<T> &values) {
            uint32_t n = values.size();
            double prob = 1. / ((double) n);
            double v = dis(gen);
            uint32_t count = 0;
            double acc = prob;
            while (v > acc) {
                count += 1;
                acc += prob;
            }

            return values[count];
        }


    private:
        std::mt19937 gen;
        std::uniform_real_distribution<double> dis;

    };

}

#endif //C___RANDOM_UTILS_H
