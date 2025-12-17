
import { businessRegistrationParser } from './src/services/ocr/parsers/businessRegistrationParser';

const testText = `
사업자등록증
(법인사업자)
등록번호 : 111-22-33333
법인명(단체명) : (주)테스트
대표자 : 홍길동
개업연월일 : 2020년 01월 01일
사업장 소재지 : 서울특별시 강남구 테헤란로 123
본 정 소 재 지 : 서울특별시 서초구 서초대로 456
사업의 종류 : 업태 서비스 종목 포털
`;

console.log("Input Text:\n", testText);

const result = businessRegistrationParser.parse(testText);

console.log("\nParsed Result:");
console.log(JSON.stringify(result, null, 2));

if (result.headAddress && result.headAddress.includes('서초구')) {
    console.log("\nSUCCESS: '본 정 소 재 지' (Spaced Typo) was correctly identified as Head Address.");
} else {
    console.log("\nFAILURE: '본 정 소 재 지' was NOT identified.");
}
