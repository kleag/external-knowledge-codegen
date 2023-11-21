// Here we test user-defined types which are supposed to be constexpr
// we use a classic kind of type which is very simple but classically constexpr

struct Position
{
    int x = 1;
    int y = 1;
};

class Vector2i
{
public:

    // TODO: use this version with `= default` once we can differenciate explicit and implicit constexpr
    // constexpr Vector2i() = default;
    constexpr Vector2i() {}
    constexpr Vector2i(int x, int y = 0) : x(x), y(y) {}

    // TODO: replace by `= default` once working (currently generates ` : x(Vector2i.x) ...`)
    constexpr Vector2i(const Vector2i& other) : x(other.x), y(other.y) {}
    constexpr Vector2i(Vector2i&& other) : x(other.x), y(other.y) {} // note: should be `x(std::move(x))` but not important here and avoid including `<utility>`

    constexpr Vector2i(const Position& position)
        : x{position.x}
        , y{position.y}
     {}

    constexpr bool operator==(const Position& position) const
    {
        return position.x == x && position.y == y;
    }

    int x = 0;
    int y = 0;
};


constexpr bool operator==(const Vector2i& left, const Vector2i& right)
{
    return left.x == right.x && left.y == right.y;
}

constexpr bool operator!=(const Vector2i& left, const Vector2i& right)
{
    return !(left == right);
}

constexpr Vector2i operator+(const Vector2i& left, const Vector2i& right)
{
    // TODO: replace by `Vector2i{ ... }` once it's not generated anymore as `Vector21( ... )` which is not equivalent
    return Vector2i( left.x + right.x, left.y + right.y );
}

constexpr Vector2i operator-(const Vector2i& left, const Vector2i& right)
{
    return Vector2i( left.x - right.x, left.y - right.y );
}

constexpr Vector2i operator*(const Vector2i& left, int scale)
{
    return Vector2i( left.x * scale, left.y * scale );
}

constexpr Vector2i operator/(const Vector2i& left, int scale)
{
    if(scale == 0) throw "invalid value"; // should fail compilation if called in constexpr context
    return Vector2i( left.x / scale, left.y / scale );
}


constexpr static auto global_object = Vector2i(42, 24);

// note: constexpr functions are implicitly inline
constexpr Vector2i some_algorithm(Vector2i vec, int value)
{
    return (vec * value) + global_object;
}

Vector2i function()
{
    constexpr Vector2i x = global_object;
    constexpr auto y = some_algorithm(x, 2);
    return y;
}

