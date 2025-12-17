
// Known dictionaries for fuzzy matching
const REGIONS = [
    '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시', '세종특별자치시',
    '경기도', '강원도', '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', '경상남도', '제주특별자치도',
    '천안시', '서북구', '동남구', '아산시', '공주시', '보령시', '서산시', '논산시', '계룡시', '당진시',
    '금산군', '부여군', '서천군', '청양군', '홍성군', '예산군', '태안군', '남양주시' // Added Namyangju
];

const KEYS = [
    { key: '등록번호', synonyms: ['등륵번호', '등록번오', '등번호'] },
    { key: '법인명', synonyms: ['법인명', '단체명', '상호', '법인명(단체명)'] },
    { key: '대표자', synonyms: ['대표자', '성명', '대표'] },
    { key: '개업연월일', synonyms: ['개업연월일', '개업일', '개업년월일'] },
    { key: '법인등록번호', synonyms: ['법인등록번호', '법인등록'] },
    { key: '소재지', synonyms: ['소재지', '사업장소재지', '본점소재지', '주소'] }
];

export interface BusinessRegistrationData {
    registrationNumber: string;
    corporateName: string;
    representative: string;
    establishmentDate: string;
    corporateRegistrationNumber: string;
    businessAddress: string;
    headAddress: string;
}

export class BusinessRegistrationParser {
    // Levenshtein Distance Implementation
    private levenshtein(a: string, b: string): number {
        const matrix = [];

        // Increment along the first column of each row
        let i;
        for (i = 0; i <= b.length; i++) {
            matrix[i] = [i];
        }

        // Increment each column in the first row
        let j;
        for (j = 0; j <= a.length; j++) {
            matrix[0][j] = j;
        }

        // Fill in the rest of the matrix
        for (i = 1; i <= b.length; i++) {
            for (j = 1; j <= a.length; j++) {
                if (b.charAt(i - 1) == a.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(matrix[i - 1][j - 1] + 1, // substitution
                        Math.min(matrix[i][j - 1] + 1, // insertion
                            matrix[i - 1][j] + 1)); // deletion
                }
            }
        }

        return matrix[b.length][a.length];
    }

    private findClosestMatch(word: string, candidates: string[], threshold: number = 2): string | null {
        let bestMatch = null;
        let minDist = Infinity;

        for (const candidate of candidates) {
            const dist = this.levenshtein(word, candidate);
            if (dist < minDist && dist <= threshold) {
                minDist = dist;
                bestMatch = candidate;
            }
        }

        return bestMatch;
    }

    // Helper: Truncate value if it runs into another key (e.g. "Name NextKey:...")
    private truncateValueAtNextKey(value: string): string {
        if (!value) return value;

        // Collect all key synonyms to watch out for
        const allSynonyms: string[] = [];
        KEYS.forEach(k => allSynonyms.push(...k.synonyms));

        // Scan the string
        // optimization: split by space and check each word? 
        // But "대표자" might be "대 표 자" attached to previous word? 
        // Let's check sliding window for safety but slow?
        // Optimized approach: Check if any known key synonym exists in the string fuzzy-wise.

        // 1. Exact match quick check
        for (const s of allSynonyms) {
            const idx = value.indexOf(s);
            if (idx > 1) { // Ignore if it's at start (shouldn't happen if we parsed correctly, but safety)
                return value.substring(0, idx).trim();
            }
        }

        // 2. Fuzzy match word by word
        const words = value.split(/\s+/);
        for (let i = 0; i < words.length; i++) {
            const word = words[i];
            // If this word looks like a Key
            if (word.length >= 2) {
                const match = this.findClosestMatch(word, allSynonyms, 1); // Strict fuzzy
                if (match) {
                    // Found a key in the middle of value!
                    // Truncate everything from this word onwards
                    // Reconstruct string up to words[i-1]
                    return words.slice(0, i).join(' ').trim();
                }
            }
        }

        return value;
    }

    private normalizeLine(line: string): string {
        // Basic cleanup
        let clean = line.replace(/\(F\)|\(f\)/g, '(주)') // Common symbol error
            .replace(/\[|\]/g, '') // Remove brackets often found in OCR
            .trim();

        // Split by space and try to correct known regions
        const words = clean.split(/\s+/);
        const correctedWords = words.map(word => {
            // Only try to correct if it looks like broken Korean (hangul) or hallucinated English
            // If it's a known region?
            const regionMatch = this.findClosestMatch(word, REGIONS, 2);
            if (regionMatch) return regionMatch;

            // Check for specific English hallucinations that user reported, mapping them to likely intended words
            // 'HOA', 'AST' -> likely '충청남도' or '서북구' is hard to guess purely by distance if the text is wildly different.
            // But if we combine Levenshtein with a small manual map for extreme outliers:
            if (word === 'HOA' || word === 'HQA') return '충청남도'; // Contextual guess? unsafe generally but requested.
            if (word === 'AST') return '서북구';
            if (word === 'AOE') return '사업장';

            return word;
        });

        return correctedWords.join(' ');
    }

    private preprocessText(text: string): string {
        // Known keys that often get merged into the previous line
        // We want to insert a newline before these if they appear in the middle of text
        const splitKeys = [
            '사업장소재지', '본점소재지', '법인등록번호', '개업연월일', '등록번호'
        ];

        let result = "";
        let i = 0;

        while (i < text.length) {
            let matchedKey: string | null = null;
            let matchLength = 0;

            // Search for keys starting at current position
            for (const key of splitKeys) {
                // Construct a candidate string by skipping spaces
                // We want 'key.length' non-space characters from text starting at 'i'
                let candidate = "";
                let scannedLength = 0;
                let k = i;

                while (candidate.length < key.length && k < text.length) {
                    const char = text[k];
                    if (!/\s/.test(char)) {
                        candidate += char;
                    }
                    k++;
                    scannedLength++; // Track how many chars we consumed in original text
                }

                // Only check if we successfully constructed a candidate of full length
                if (candidate.length === key.length) {
                    const dist = this.levenshtein(candidate, key);
                    // Threshold: allows 1 error per 4-5 chars
                    if (dist <= 1) {
                        matchedKey = key;
                        matchLength = scannedLength; // Consume the original length (including spaces)
                        break;
                    }
                }
            }

            if (matchedKey) {
                // Found a key (or its fuzzy variation)
                // If the previous character in our result is not a newline, add one.
                const lastChar = result.length > 0 ? result[result.length - 1] : '';

                if (lastChar !== '\n' && lastChar !== '\r' && result.length > 0) {
                    result += '\n';
                }

                // Append the matched text from the original string
                result += text.substring(i, i + matchLength);
                i += matchLength;
            } else {
                result += text[i];
                i++;
            }
        }

        return result;
    }

    parse(text: string): BusinessRegistrationData {
        // Step 0: Preprocess text to split merged lines
        const cleanText = this.preprocessText(text);

        // Robust splitting for various newline formats
        const lines = cleanText.split(/\r\n|\n|\r/)
            .map(line => this.normalizeLine(line))
            .filter(line => line.length > 0);

        const data: BusinessRegistrationData = {
            registrationNumber: '',
            corporateName: '',
            representative: '',
            establishmentDate: '',
            corporateRegistrationNumber: '',
            businessAddress: '',
            headAddress: ''
        };

        // Two-Pass Parsing Strategy
        // Pass 1: Strict (Threshold 1) - Safe, low false positives
        // Pass 2: Loose (Threshold 2) - Catch-all for heavy noise (e.g. "멈인영")
        const thresholds = [1, 2];

        for (const threshold of thresholds) {
            for (const line of lines) {
                // Normalize colons and spaces for key matching
                const normalizedLine = line.replace(/：/g, ':');
                const colonParts = normalizedLine.split(':');
                const hasColon = colonParts.length > 1;

                // "Clean" the key part by removing spaces AND punctuation/symbols
                // Keep only Hangul, English, and Digits.
                const cleanKey = hasColon
                    ? colonParts[0].replace(/[^가-힣a-zA-Z0-9]/g, '')
                    : normalizedLine.replace(/[^가-힣a-zA-Z0-9]/g, '');

                const parts = hasColon ? colonParts.slice(1).join(':').trim().split(/\s{2,}/) : [];
                const cleanedValue = parts.length > 0 ? parts[0].trim() : '';

                const valuePart = hasColon ? colonParts.slice(1).join(':').trim() : '';

                // Fuzzy match helper
                const isMatch = (targets: string[]) => {
                    // 1. Exact inclusion (fastest)
                    if (targets.some(t => cleanKey.includes(t))) return true;

                    for (const target of targets) {
                        // 2. Full key fuzzy match
                        if (this.levenshtein(cleanKey, target) <= threshold) return true;

                        // 3. Prefix fuzzy match
                        if (cleanKey.length >= target.length) {
                            const prefix = cleanKey.substring(0, target.length);
                            if (this.levenshtein(prefix, target) <= threshold) return true;
                        }
                        if (cleanKey.length >= target.length + 1) {
                            const prefixPlus = cleanKey.substring(0, target.length + 1);
                            if (this.levenshtein(prefixPlus, target) <= threshold) return true;
                        }
                    }
                    return false;
                };

                // 1. Registration Number
                if (!data.registrationNumber) {
                    if (isMatch(['등록번호']) || normalizedLine.match(/\d{3}-\d{2}-\d{5}/)) {
                        const match = normalizedLine.match(/\d{3}[-\s]?\d{2}[-\s]?\d{5}/);
                        if (match) data.registrationNumber = match[0];
                    }
                }

                // 2. Corporate Name
                // 2. Corporate Name
                // Fix: Exclude '법인사업' lines (header) to avoid matching '법인명' via fuzzy prefix
                // e.g. "법인사업ㅅ" starts with "법인사업" and should be ignored.
                if (!data.corporateName && !cleanKey.startsWith('법인사업') && !cleanKey.startsWith('사업자') && isMatch(['법인명', '단체명', '상호'])) {
                    let rawVal = cleanedValue;
                    if (!rawVal) {
                        rawVal = normalizedLine.replace(/법인명|단체명|\(단체명\)|상호|:/g, '').trim();
                    }
                    data.corporateName = this.truncateValueAtNextKey(rawVal.replace(/\(법인명\)|\(단체명\)/g, '')).trim();
                }

                // 3. Representative
                if (!data.representative && isMatch(['대표자', '성명'])) {
                    let finalValue = cleanedValue;

                    // Same-line noise correction (e.g. "Name Region")
                    if (finalValue) {
                        for (const region of REGIONS) {
                            const index = finalValue.indexOf(region);
                            if (index !== -1) {
                                finalValue = finalValue.substring(0, index).trim();
                                break;
                            }
                        }
                    }

                    if (finalValue) {
                        data.representative = this.truncateValueAtNextKey(finalValue);
                    } else {
                        // Fallback logic
                        let temp = normalizedLine.replace(/[^가-힣a-zA-Z0-9\s]/g, ' ').trim();
                        temp = temp.replace(/대표자|내표자|성명/g, '').trim();
                        for (const region of REGIONS) {
                            const index = temp.indexOf(region);
                            if (index !== -1) temp = temp.substring(0, index).trim();
                        }
                        data.representative = this.truncateValueAtNextKey(temp);
                    }
                }

                // 4. Establishment Date
                if (!data.establishmentDate && (isMatch(['개업연월일', '개업일']))) {
                    const match = normalizedLine.match(/\d{4}\s?년\s?\d{2}\s?월\s?\d{2}\s?일/);
                    if (match) {
                        data.establishmentDate = match[0];
                    } else if (cleanedValue) {
                        data.establishmentDate = this.truncateValueAtNextKey(cleanedValue);
                    }
                }

                // 5. Corporate Registration Number
                if (!data.corporateRegistrationNumber && normalizedLine.replace(/\s+/g, '').includes('법인등록번호')) {
                    const match = normalizedLine.match(/\d{6}[-\s]?\d{7}/);
                    if (match) data.corporateRegistrationNumber = match[0];
                }

                // 6. Address
                if (isMatch(['소재지', '사업장', '주소'])) {
                    // Fix: Use fuzzy match for '본점' too, not just strict include.
                    // cleanKey can be '본정소재지'. '본점' is not in '본정소재지'.
                    // We check if '본점' or '본점소재지' is close to cleanKey.

                    const isHeadOffice = (
                        cleanKey.includes('본점') ||
                        this.levenshtein(cleanKey.substring(0, 4), '본점소재') <= 1 ||
                        this.levenshtein(cleanKey, '본점소재지') <= 2 ||
                        cleanKey.includes('본정') // Explicit known typo fallback
                    );

                    const isBusinessPlace = cleanKey.includes('사업장');

                    let addressValue = valuePart; // Address can be long and contain spaces
                    if (!addressValue && REGIONS.some(r => normalizedLine.includes(r))) {
                        addressValue = normalizedLine.replace(/본점소재지|사업장소재지|소재지|사업장|본점|주소|:/g, '').trim();
                    }

                    if (addressValue) {
                        if (isHeadOffice && !data.headAddress) {
                            data.headAddress = addressValue;
                        } else if (!data.businessAddress) {
                            if (!isHeadOffice || isBusinessPlace) {
                                data.businessAddress = addressValue;
                            }
                        }
                    }
                }
            }
        }

        // Final fallback for businessAddress if empty
        if (!data.businessAddress && !data.headAddress) {
            for (const line of lines) {
                for (const region of REGIONS) {
                    if (line.includes(region)) {
                        data.businessAddress = line.replace(/본점소재지|사업장소재지|소재지|:/g, '').trim();
                        break;
                    }
                }
                if (data.businessAddress) break;
            }
        }

        return data;
    }
}

export const businessRegistrationParser = new BusinessRegistrationParser();
