using Test
using differential

@testset "Basic tests" begin
    @test 1 + 1 == 2
    @test reverse([1, 2, 3]) == [3, 2, 1]
end
