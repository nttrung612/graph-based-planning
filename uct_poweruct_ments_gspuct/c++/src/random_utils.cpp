#include <random_utils.h>

namespace poweruct {

    RandomUtils::RandomUtils() : gen(), dis(0., 1.) {}

    RandomUtils::~RandomUtils() {}

    void RandomUtils::seed(uint32_t s) {
        gen.seed(s);
    }

    uint32_t RandomUtils::sample_discrete(std::vector<double> &probabilities) {
        double v = dis(gen);
        uint32_t count = 0;
        double acc = probabilities[0];
        while (v > acc) {
            count += 1;
            acc += probabilities[count];
        }

        return count;
    }

    double RandomUtils::sample_uniform() {
        return dis(gen);
    }

    uint32_t RandomUtils::sample_uniform(uint32_t n) {
        double prob = 1. / ((double) n);
        double v = dis(gen);

        uint32_t count = 0;
        double acc = prob;
        while (v > acc) {
            count += 1;
            acc += prob;
        }

        return count;
    }
}