
const REGIONS = [
    '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시', '세종특별자치시',
    '경기도', '강원도', '충청북도', '충청남도'
];

class BusinessRegistrationParser {
    public levenshtein(a: string, b: string): number {
        const matrix = [];
        let i;
        for (i = 0; i <= b.length; i++) {
            matrix[i] = [i];
        }
        let j;
        for (j = 0; j <= a.length; j++) {
            matrix[0][j] = j;
        }
        for (i = 1; i <= b.length; i++) {
            for (j = 1; j <= a.length; j++) {
                if (b.charAt(i - 1) == a.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(matrix[i - 1][j - 1] + 1,
                        Math.min(matrix[i][j - 1] + 1,
                            matrix[i - 1][j] + 1));
                }
            }
        }
        return matrix[b.length][a.length];
    }

    public preprocessText(text: string): string {
        const splitKeys = [
            '사업장소재지', '본점소재지', '법인등록번호', '개업연월일', '등록번호'
        ];

        let result = "";
        let i = 0;

        console.log(`Analyzing text length: ${text.length}`);

        while (i < text.length) {
            let matchedKey: string | null = null;
            let matchLength = 0;

            // Verbose logging for '본'
            if (text[i] === '본') {
                console.log(`At index ${i}, char is '본'. Checking keys...`);
            }

            for (const key of splitKeys) {
                const sub = text.substring(i, i + key.length);

                if (text[i] === '본' && key === '본점소재지') {
                    const d = this.levenshtein(sub, key);
                    console.log(`  Comparing sub '${sub}' (len ${sub.length}) with key '${key}' -> Dist: ${d}`);
                }

                if (sub.length === key.length) {
                    const dist = this.levenshtein(sub, key);
                    if (dist <= 1) {
                        // Double check: if it is length 4, dist 1 is 25% error.
                        // But these keys are mostly 5+ chars. '등록번호' is 4.
                        matchedKey = key;
                        matchLength = key.length;
                        break;
                    }
                }
            }

            if (matchedKey) {
                console.log(`Matched key '${matchedKey}' at index ${i}`);
                const lastChar = result.length > 0 ? result[result.length - 1] : '';

                if (lastChar !== '\n' && lastChar !== '\r' && result.length > 0) {
                    result += '\n';
                }

                result += text.substring(i, i + matchLength);
                i += matchLength;
            } else {
                result += text[i];
                i++;
            }
        }

        return result;
    }
}

const parser = new BusinessRegistrationParser();
// The string exactly as noticed in the raw text
const rawText = "사업장소재지:충청남도천안시서북구업성3길168(신당동)본정소재지:충청남도천안시서북구업성3길168(신당동)";
const processed = parser.preprocessText(rawText);

console.log("--- Result ---");
console.log(JSON.stringify(processed));
console.log("--- End Result ---");
