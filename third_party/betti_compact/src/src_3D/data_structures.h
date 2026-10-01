#pragma once

#include "../config.h"

#include <algorithm>
#include <cstddef> // For std::ptrdiff_t
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <iostream>
#include <iterator> // For std::forward_iterator_tag
#include <optional>
#include <tuple>
#include <vector>

#include <sstream>

using namespace std;

namespace dim3 {
typedef tuple<index_t, index_t, index_t> Coordinate;
typedef vector<Coordinate> RepresentativeCycle;

class Cube {
  public:
    static constexpr index_t kCoordinateLimit = 512;
    static constexpr uint64_t kCoordinateMask = kCoordinateLimit - 1;
    static constexpr uint64_t kZShift = 2;
    static constexpr uint64_t kYShift = kZShift + 9;
    static constexpr uint64_t kXShift = kYShift + 9;
    static constexpr uint64_t kBirthShift = 30;
    static constexpr uint64_t kIdentityMask = (uint64_t{1} << kBirthShift) - 1;

    Cube();
    Cube(value_t birth, index_t x, index_t y, index_t z, uint8_t type);
    Cube(value_t birth, vector<index_t> coordinates, uint8_t type);
    Cube(const Cube &cube);
    bool operator==(const Cube &rhs) const;
    index_t x() const;
    index_t y() const;
    index_t z() const;
    uint8_t type() const;
    uint64_t identity() const;
    value_t filtration() const;
    void clearIdentity();
    void print() const;

  private:
    uint32_t packed;
};

static_assert(sizeof(Cube) == sizeof(uint32_t),
              "Binary Cube must pack identity and birth into one word");

struct CubeComparator {
    bool operator()(const Cube &Cube1, const Cube &Cube2) const;
};

class Pair {
  public:
    Pair();
    Pair(const Cube &birth, const Cube &death);
    Pair(const Pair &pair);
    bool operator==(const Pair &rhs) const;
    void print() const;
    Cube birth;
    Cube death;
};

static_assert(sizeof(Pair) == 2 * sizeof(Cube),
              "Binary Pair must contain two packed cubes");

class Match {
  public:
    Match(Pair pair0, Pair pair1);
    void print() const;
    Pair pair0;
    Pair pair1;
};

static_assert(sizeof(Match) == 2 * sizeof(Pair),
              "Binary Match must contain two compact pairs");

class CubicalGridComplex {
  public:
    CubicalGridComplex(const vector<value_t> &image,
                       const vector<index_t> &shape);
    CubicalGridComplex(CubicalGridComplex &&other);
    ~CubicalGridComplex();
    size_t getNumberOfCubes(const uint8_t &dim) const;
    value_t getBirth(const index_t &x, const index_t &y,
                     const index_t &z) const;
    value_t getBirth(const index_t &x, const index_t &y, const index_t &z,
                     const uint8_t &type, const uint8_t &dim) const;
    Coordinate getParentVoxel(const Cube &c, const uint8_t &dim) const;
    void printImage() const;
    void printRepresentativeCycle(const RepresentativeCycle &reprCycle) const;
    const vector<index_t> shape;
    const index_t m_x;
    const index_t m_y;
    const index_t m_z;
    const index_t m_yz;
    const index_t m_xyz;
    const index_t n_yz;
    const index_t n_xyz;

  private:
    value_t ***allocateMemory() const;
    void getGridFromVector(const vector<value_t> &vector);
    value_t ***grid;
};

class UnionFind {
  public:
    UnionFind(const CubicalGridComplex &cgc);
    index_t find(index_t x);
    index_t link(index_t x, index_t y);
    value_t getBirth(const index_t &idx) const;
    Coordinate getCoordinates(index_t idx) const;
    vector<index_t> getBoundaryIndices(const Cube &edge) const;
    void reset();

  private:
    vector<index_t> parent;
    vector<value_t> birthtime;
    const CubicalGridComplex &cgc;
};

class UnionFindDual {
  public:
    UnionFindDual(const CubicalGridComplex &cgc);
    index_t find(index_t x);
    index_t link(index_t x, index_t y);
    value_t getBirth(const index_t &idx) const;
    Coordinate getCoordinates(index_t idx) const;
    vector<index_t> getBoundaryIndices(const Cube &edge) const;
    void reset();

  private:
    vector<index_t> parent;
    vector<value_t> birthtime;
    const CubicalGridComplex &cgc;
};

// Dense identities retain O(1) lookup, but absent entries cost one uint32.
// Keep payload references stable during pool growth and preserve optional
// assignment/reset semantics used by the upstream replay cache.
template <int Dim> class CubeMapIndex {
  protected:
    explicit CubeMapIndex(const vector<index_t> &shape)
        : shape(shape), strideX(size_t(shape[1]) * shape[2] * NUM_TYPES),
          strideY(size_t(shape[2]) * NUM_TYPES) {}
    size_t slotCount() const {
        return size_t(shape[0]) * shape[1] * shape[2] * NUM_TYPES;
    }
    size_t slot(uint64_t identity) const {
        const size_t x = (identity >> Cube::kXShift) & Cube::kCoordinateMask;
        const size_t y = (identity >> Cube::kYShift) & Cube::kCoordinateMask;
        const size_t z = (identity >> Cube::kZShift) & Cube::kCoordinateMask;
        const size_t type = NUM_TYPES == 3 ? identity & 3 : 0;
        return x * strideX + y * strideY + z * NUM_TYPES + type;
    }
    vector<index_t> shape;
    const size_t strideX, strideY;
    static constexpr size_t NUM_TYPES = (Dim == 1 || Dim == 2) ? 3 : 1;
};

template <int Dim, class T, class Enable = void> class CubeMap
    : private CubeMapIndex<Dim> {
  public:
    explicit CubeMap(vector<index_t> shape)
        : CubeMapIndex<Dim>(shape), handles(this->slotCount(), 0) {}
    void emplace(uint64_t identity, T value) {
        (*this)[identity] = std::move(value);
    }
    const optional<T> &find(uint64_t identity) const {
        if (identity == NONE_INDEX) return none;
        const uint32_t handle = handles[this->slot(identity)];
        return handle == 0 ? none : payloads[handle - 1];
    }
    optional<T> &operator[](uint64_t identity) {
        if (identity == NONE_INDEX)
            throw runtime_error("CubeMap subscript may not use NONE_INDEX");
        uint32_t &handle = handles[this->slot(identity)];
        if (handle == 0) {
            if (payloads.size() >= numeric_limits<uint32_t>::max())
                throw overflow_error("CubeMap payload handle overflow");
            payloads.emplace_back();
            handle = static_cast<uint32_t>(payloads.size());
        }
        return payloads[handle - 1];
    }
    void clear() {
        std::fill(handles.begin(), handles.end(), 0);
        payloads.clear();
    }
  private:
    vector<uint32_t> handles;
    deque<optional<T>> payloads;
    optional<T> none;
};

// Pivot-column numbers and packed identities fit uint32 for the supported
// input sizes. Return a small optional by value; no optional object is stored
// per cell. The sentinel is checked instead of silently truncating an index.
template <int Dim, class T>
class CubeMap<Dim, T, enable_if_t<is_integral_v<T>>>
    : private CubeMapIndex<Dim> {
  public:
    explicit CubeMap(vector<index_t> shape)
        : CubeMapIndex<Dim>(shape), elements(this->slotCount(), EMPTY) {}
    void emplace(uint64_t identity, T value) {
        if (identity == NONE_INDEX)
            throw runtime_error("CubeMap::emplace may not use NONE_INDEX");
        if (uint64_t(value) >= EMPTY)
            throw overflow_error("CubeMap scalar exceeds uint32 storage");
        elements[this->slot(identity)] = static_cast<uint32_t>(value);
    }
    optional<T> find(uint64_t identity) const {
        if (identity == NONE_INDEX) return nullopt;
        const uint32_t value = elements[this->slot(identity)];
        return value == EMPTY ? nullopt : optional<T>(static_cast<T>(value));
    }
    class Reference {
      public:
        Reference(CubeMap &map, uint64_t identity) : map(map), identity(identity) {}
        Reference &operator=(T value) { map.emplace(identity, value); return *this; }
      private:
        CubeMap &map;
        uint64_t identity;
    };
    Reference operator[](uint64_t identity) { return Reference(*this, identity); }
    void clear() { std::fill(elements.begin(), elements.end(), EMPTY); }
  private:
    static constexpr uint32_t EMPTY = numeric_limits<uint32_t>::max();
    vector<uint32_t> elements;
};
} // namespace dim3
