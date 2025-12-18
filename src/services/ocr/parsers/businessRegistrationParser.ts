
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
    // Cache for levenshtein results to avoid redundant calculations
    private levenshteinCache: Map<string, number> = new Map();

    // Pre-computed Set for O(1) region lookup
    private regionSet: Set<string> = new Set(REGIONS);

    // Clear cache between parse calls to prevent memory growth
    private clearCache(): void {
        this.levenshteinCache.clear();
    }

    // Levenshtein Distance with Caching and Early Return Optimization
    private levenshtein(a: string, b: string, maxThreshold: number = 3): number {
        // Early return: identical strings
        if (a === b) return 0;

        // Early return: length difference exceeds threshold
        const lengthDiff = Math.abs(a.length - b.length);
        if (lengthDiff > maxThreshold) return lengthDiff;

        // Early return: empty strings
        if (a.length === 0) return b.length;
        if (b.length === 0) return a.length;

        // Check cache
        const cacheKey = a < b ? `${a}|${b}` : `${b}|${a}`;
        const cached = this.levenshteinCache.get(cacheKey);
        if (cached !== undefined) return cached;

        // Optimized: Use single array instead of full matrix (space O(min(n,m)))
        const shorter = a.length < b.length ? a : b;
        const longer = a.length < b.length ? b : a;
        const m = shorter.length;
        const n = longer.length;

        let prevRow = new Array(m + 1);
        let currRow = new Array(m + 1);

        // Initialize first row
        for (let j = 0; j <= m; j++) {
            prevRow[j] = j;
        }

        // Fill in the matrix row by row
        for (let i = 1; i <= n; i++) {
            currRow[0] = i;

            // Early termination: if minimum possible distance exceeds threshold
            let rowMin = currRow[0];

            for (let j = 1; j <= m; j++) {
                if (longer.charAt(i - 1) === shorter.charAt(j - 1)) {
                    currRow[j] = prevRow[j - 1];
                } else {
                    currRow[j] = Math.min(
                        prevRow[j - 1] + 1,  // substitution
                        prevRow[j] + 1,      // deletion
                        currRow[j - 1] + 1   // insertion
                    );
                }
                rowMin = Math.min(rowMin, currRow[j]);
            }

            // Early termination if all values in row exceed threshold
            if (rowMin > maxThreshold) {
                this.levenshteinCache.set(cacheKey, rowMin);
                return rowMin;
            }

            // Swap rows
            [prevRow, currRow] = [currRow, prevRow];
        }

        const result = prevRow[m];
        this.levenshteinCache.set(cacheKey, result);
        return result;
    }

    private findClosestMatch(word: string, candidates: string[], threshold: number = 2): string | null {
        // Fast path: exact match check O(1) if using Set
        if (candidates === REGIONS && this.regionSet.has(word)) {
            return word;
        }

        // Check cache-friendly exact match first
        for (const candidate of candidates) {
            if (word === candidate) return candidate;
        }

        // Only do expensive fuzzy matching if word length is similar to candidates
        let bestMatch = null;
        let minDist = threshold + 1;  // Start above threshold

        for (const candidate of candidates) {
            // Skip if length difference makes match impossible
            if (Math.abs(word.length - candidate.length) > threshold) continue;

            const dist = this.levenshtein(word, candidate, threshold);
            if (dist < minDist) {
                minDist = dist;
                bestMatch = candidate;
                // Early exit if exact match found (dist = 0)
                if (dist === 0) break;
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
            // Fast path: O(1) exact match using Set
            if (this.regionSet.has(word)) return word;

            // Check for specific English hallucinations (known mappings)
            // These bypass expensive fuzzy matching
            switch (word) {
                case 'HOA': case 'HQA': return '충청남도';
                case 'AST': return '서북구';
                case 'AOE': return '사업장';
            }

            // Only try fuzzy matching for short words that look like they might be regions
            // Skip if word doesn't contain hangul or is too short/long for a region name
            if (word.length >= 2 && word.length <= 6 && /[가-힯]/.test(word)) {
                const regionMatch = this.findClosestMatch(word, REGIONS, 2);
                if (regionMatch) return regionMatch;
            }

            return word;
        });

        return correctedWords.join(' ');
    }

    /**
     * Levenshtein + Regex Hybrid Approach for Field Detection
     * 
     * Strategy:
     * 1. Regex: Find all potential "키워드:값" or "키워드 :" patterns
     * 2. Levenshtein: Fuzzy match extracted keywords against known field names
     * 3. Insert newlines before matched field starts
     * 
     * This avoids false positives by requiring structural context (colon)
     */

    // Known canonical field keywords (NO typo variations - Levenshtein handles those)
    private readonly FIELD_KEYWORDS = [
        '사업장소재지',
        '본점소재지',
        '법인등록번호',
        '개업연월일',
        '등록번호',
        '대표자',
        '법인명',
        '단체명'
    ];

    /**
     * Check if a candidate string fuzzy-matches any known field keyword
     * Handles garbage prefixes like "일법인등록번호" → "법인등록번호"
     * @returns The matched keyword if found, null otherwise
     */
    private fuzzyMatchFieldKeyword(candidate: string, maxDistance: number = 2): string | null {
        // Normalize candidate
        const normalizedCandidate = candidate.replace(/\s+/g, '').trim();

        // Try matching with different prefix removals (0, 1, 2 chars)
        const candidatesToTry = [
            normalizedCandidate,                           // Original
            normalizedCandidate.substring(1),              // Remove 1 char prefix
            normalizedCandidate.substring(2),              // Remove 2 char prefix
        ].filter(c => c.length >= 2); // Only try if result has at least 2 chars

        for (const tryCandidate of candidatesToTry) {
            for (const keyword of this.FIELD_KEYWORDS) {
                // Exact match (fast path)
                if (tryCandidate === keyword) return keyword;

                // Check if candidate contains keyword
                if (tryCandidate.includes(keyword)) return keyword;

                // Levenshtein fuzzy match
                // Only compare if lengths are reasonably similar
                const lenDiff = Math.abs(tryCandidate.length - keyword.length);
                if (lenDiff <= maxDistance) {
                    const distance = this.levenshtein(tryCandidate, keyword, maxDistance);
                    if (distance <= maxDistance) return keyword;
                }

                // Check if candidate is a substring match with OCR errors
                // e.g., "법인들록번호" should match "법인등록번호"
                if (tryCandidate.length >= keyword.length) {
                    const candidateSubstr = tryCandidate.substring(0, keyword.length);
                    const distance = this.levenshtein(candidateSubstr, keyword, maxDistance);
                    if (distance <= maxDistance) return keyword;
                }
            }
        }

        return null;
    }

    private preprocessText(text: string): string {
        let result = text;

        // =================================================================
        // PASS 0: Separate date suffixes from following Korean text
        // =================================================================
        // Fixes: "2015년12월01일법" -> "2015년12월01일 법"
        // Where "법" is actually the start of "법인등록번호" but OCR merged them
        result = result.replace(/(\d{1,2}일)([가-힣])/g, '$1 $2');

        // =================================================================
        // PASS 0.5: Fix OCR space within "법인등록번호" keyword
        // =================================================================
        // Fixes: "법인등록번 호:" -> "법인등록번호:"
        // OCR sometimes inserts a space before "호"
        result = result.replace(/등록번\s+호\s*[:：]/g, '등록번호:');

        // =================================================================
        // PASS 1: Regex to find "키워드:값" patterns and split them
        // =================================================================
        // This regex finds patterns like "키워드:" where 키워드 is 2-8 Korean chars
        // It captures the position to potentially insert newlines

        // Pattern: Korean word (2-8 chars) followed by colon
        const keyValuePattern = /([가-힣]{2,8})\s*[:：]/g;

        const matches: { index: number; key: string; fullMatch: string }[] = [];
        let match;

        while ((match = keyValuePattern.exec(result)) !== null) {
            matches.push({
                index: match.index,
                key: match[1],
                fullMatch: match[0]
            });
        }

        // =================================================================
        // PASS 2: Levenshtein to check if each matched key is a known field
        // =================================================================
        const boundariesToInsert: number[] = [];

        for (const m of matches) {
            const matchedKeyword = this.fuzzyMatchFieldKeyword(m.key, 2);

            if (matchedKeyword) {
                // This looks like a field - mark for newline insertion
                // But only if not at start and not already preceded by newline
                if (m.index > 0 && result[m.index - 1] !== '\n' && result[m.index - 1] !== '\r') {
                    boundariesToInsert.push(m.index);
                }
            }
        }

        // =================================================================
        // PASS 3: Handle special OCR garbage prefixes
        // =================================================================
        // Patterns like "일법인등록번호", "일번인들록번호" where "일" is garbage
        const garbagePrefixPattern = /([일법번])([가-힣]{2,7})[:：]/g;

        while ((match = garbagePrefixPattern.exec(result)) !== null) {
            const potentialKeyword = match[2];
            const matchedKeyword = this.fuzzyMatchFieldKeyword(potentialKeyword, 2);

            if (matchedKeyword) {
                // Insert newline before the garbage prefix
                if (match.index > 0 && result[match.index - 1] !== '\n' && result[match.index - 1] !== '\r') {
                    if (!boundariesToInsert.includes(match.index)) {
                        boundariesToInsert.push(match.index);
                    }
                }
            }
        }

        // =================================================================
        // PASS 4: Insert newlines at boundaries (work backwards)
        // =================================================================
        // Sort descending to preserve indices when inserting
        boundariesToInsert.sort((a, b) => b - a);

        for (const idx of boundariesToInsert) {
            result = result.substring(0, idx) + '\n' + result.substring(idx);
        }

        return result;
    }

    parse(text: string): BusinessRegistrationData {
        // Clear cache at start of each parse to prevent memory growth
        this.clearCache();

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

        // Pre-compiled regex patterns for fast matching (replaces expensive levenshtein in isMatch)
        const keyPatterns = {
            registrationNumber: /등[록녹][번빈][호호]|홍록번호/,
            corporateName: /법인[명명]|단체[명명]|상호/,
            representative: /대[표표][자자]|성[명명]/,
            establishmentDate: /개[업엽][연년][월웕]일|개[업엽]일/,
            address: /소[재제]지|사업[장쟝]|주소/,
            headOffice: /[본분][점정]/,
        };

        // Single-pass parsing (optimized from two-pass)
        // Uses regex for primary matching, falls back to levenshtein only when needed
        for (const line of lines) {
            // Normalize colons and spaces for key matching
            const normalizedLine = line.replace(/：/g, ':');
            const colonParts = normalizedLine.split(':');
            const hasColon = colonParts.length > 1;

            // "Clean" the key part by removing spaces AND punctuation/symbols
            // Keep only Hangul, English, and Digits.
            const cleanKey = hasColon
                ? colonParts[0].replace(/[^가-힯a-zA-Z0-9]/g, '')
                : normalizedLine.replace(/[^가-힯a-zA-Z0-9]/g, '');

            const parts = hasColon ? colonParts.slice(1).join(':').trim().split(/\s{2,}/) : [];
            const cleanedValue = parts.length > 0 ? parts[0].trim() : '';

            const valuePart = hasColon ? colonParts.slice(1).join(':').trim() : '';

            // Optimized matching: Use regex first, only fallback to levenshtein for edge cases
            const isMatchFast = (pattern: RegExp, exactTargets: string[]) => {
                // 1. Regex match (fastest)
                if (pattern.test(cleanKey)) return true;

                // 2. Exact inclusion check
                if (exactTargets.some(t => cleanKey.includes(t))) return true;

                return false;
            };

            // 1. Registration Number
            if (!data.registrationNumber) {
                if (isMatchFast(keyPatterns.registrationNumber, ['등록번호']) || normalizedLine.match(/\d{3}-\d{2}-\d{5}/)) {
                    const match = normalizedLine.match(/\d{3}[-\s]?\d{2}[-\s]?\d{5}/);
                    if (match) data.registrationNumber = match[0];
                }
            }

            // 2. Corporate Name
            // Fix: Exclude '법인사업' lines (header) to avoid matching '법인명' via fuzzy prefix
            if (!data.corporateName && !cleanKey.startsWith('법인사업') && !cleanKey.startsWith('사업자') && isMatchFast(keyPatterns.corporateName, ['법인명', '단체명', '상호'])) {
                let rawVal = cleanedValue;
                if (!rawVal) {
                    rawVal = normalizedLine.replace(/법인명|단체명|\(단체명\)|상호|:/g, '').trim();
                }
                data.corporateName = this.truncateValueAtNextKey(rawVal.replace(/\(법인명\)|\(단체명\)/g, '')).trim();
            }

            // 3. Representative
            if (!data.representative && isMatchFast(keyPatterns.representative, ['대표자', '성명'])) {
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
                    let temp = normalizedLine.replace(/[^가-힯a-zA-Z0-9\s]/g, ' ').trim();
                    temp = temp.replace(/대표자|내표자|성명/g, '').trim();
                    for (const region of REGIONS) {
                        const index = temp.indexOf(region);
                        if (index !== -1) temp = temp.substring(0, index).trim();
                    }
                    data.representative = this.truncateValueAtNextKey(temp);
                }
            }

            // 4. Establishment Date
            if (!data.establishmentDate && isMatchFast(keyPatterns.establishmentDate, ['개업연월일', '개업일'])) {
                const match = normalizedLine.match(/\d{4}\s?년\s?\d{2}\s?월\s?\d{2}\s?일/);
                if (match) {
                    data.establishmentDate = match[0];
                } else if (cleanedValue) {
                    data.establishmentDate = this.truncateValueAtNextKey(cleanedValue);
                }
            }

            // 5. Corporate Registration Number
            // Optimized: Use regex pattern matching instead of sliding window + levenshtein
            if (!data.corporateRegistrationNumber) {
                const lineNoSpaces = normalizedLine.replace(/\s+/g, '');

                // Regex for OCR typo variations of 법인등록번호
                // [등들둥] handles 등→들/둥 typos, [록녹륙] handles 록→녹/륙 typos
                const corpRegNumPattern = /법인[등들둥][록녹륙][번빈]호/;
                const foundFuzzyMatch = corpRegNumPattern.test(lineNoSpaces);

                if (foundFuzzyMatch) {
                    // Try multiple regex patterns for the number itself
                    // Standard: 6-7 digits, but OCR may produce variations
                    const patterns = [
                        /[:：]\s*(\d{6})[-\s]?(\d{7})/,   // After colon: 6-7 format
                        /(\d{6})[-\s]?(\d{7})/,           // Standard 6-7 format
                        /(\d{6})[^\d]?(\d{7})/,           // 6-7 with any separator
                        /(\d{13})/,                       // 13 consecutive digits
                    ];

                    for (const pattern of patterns) {
                        const match = normalizedLine.match(pattern);
                        if (match) {
                            if (match[2]) {
                                data.corporateRegistrationNumber = match[1] + '-' + match[2];
                            } else if (match[1] && match[1].length === 13) {
                                data.corporateRegistrationNumber = match[1].substring(0, 6) + '-' + match[1].substring(6);
                            }
                            break;
                        }
                    }
                }
            }

            // 6. Address
            if (isMatchFast(keyPatterns.address, ['소재지', '사업장', '주소'])) {
                // Use regex for head office detection instead of levenshtein
                const isHeadOffice = (
                    keyPatterns.headOffice.test(cleanKey) ||
                    cleanKey.includes('본점') ||
                    cleanKey.includes('본정')
                );

                const isBusinessPlace = cleanKey.includes('사업장');

                let addressValue = valuePart;
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
