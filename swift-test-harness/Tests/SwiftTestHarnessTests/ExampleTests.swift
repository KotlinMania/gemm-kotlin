#if canImport(Testing)
import Testing
import Gemm

@Suite("Gemm Swift Export Tests")
struct GemmExportTests {
    @Test("Gemm swift module imported cleanly")
    func testSwiftModuleLoads() throws {
        #expect(Bool(true), "Gemm swift module imported cleanly")
    }
}
#elseif canImport(XCTest)
import XCTest
import Gemm

final class GemmExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        XCTAssertTrue(true, "Gemm swift module imported cleanly")
    }
}
#endif
