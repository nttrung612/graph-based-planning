//
// Created by klink on 31.07.19.
//

#ifndef C___UTIL_H
#define C___UTIL_H


#include <memory>

namespace poweruct {


    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }


};


#endif //C___UTIL_H
