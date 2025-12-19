/**
 * 한글 자소 분리/조합 유틸리티
 * 
 * 한글 유니코드 구조:
 * 완성형 한글 = 0xAC00 + (초성 * 588) + (중성 * 28) + 종성
 * 
 * 범위: 가(0xAC00) ~ 힣(0xD7A3)
 */

// 초성 19자
export const CHO = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
];

// 중성 21자
export const JUNG = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
];

// 종성 28자 (첫 번째는 종성 없음)
export const JONG = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
];

const HANGUL_START = 0xAC00;
const HANGUL_END = 0xD7A3;

export interface HangulJamo {
    cho: string;   // 초성
    jung: string;  // 중성
    jong: string;  // 종성
}

/**
 * 한글 문자인지 확인
 */
export function isHangul(char: string): boolean {
    if (char.length !== 1) return false;
    const code = char.charCodeAt(0);
    return code >= HANGUL_START && code <= HANGUL_END;
}

/**
 * 한글 자모인지 확인 (ㄱ-ㅎ, ㅏ-ㅣ)
 */
export function isJamo(char: string): boolean {
    if (char.length !== 1) return false;
    const code = char.charCodeAt(0);
    return (code >= 0x3131 && code <= 0x3163);
}

/**
 * 한글 완성형 문자를 자소로 분리
 */
export function decomposeHangul(char: string): HangulJamo | null {
    if (!isHangul(char)) return null;

    const code = char.charCodeAt(0) - HANGUL_START;
    const choIdx = Math.floor(code / 588);
    const jungIdx = Math.floor((code % 588) / 28);
    const jongIdx = code % 28;

    return {
        cho: CHO[choIdx],
        jung: JUNG[jungIdx],
        jong: JONG[jongIdx]
    };
}

/**
 * 자소를 조합하여 완성형 한글로
 */
export function composeHangul(cho: string, jung: string, jong: string = ''): string | null {
    const choIdx = CHO.indexOf(cho);
    const jungIdx = JUNG.indexOf(jung);
    const jongIdx = JONG.indexOf(jong);

    if (choIdx === -1 || jungIdx === -1 || jongIdx === -1) {
        return null;
    }

    const code = HANGUL_START + (choIdx * 588) + (jungIdx * 28) + jongIdx;
    return String.fromCharCode(code);
}

/**
 * 문자열의 모든 한글을 자소로 분리
 */
export function decomposeString(text: string): string {
    let result = '';
    for (const char of text) {
        const jamo = decomposeHangul(char);
        if (jamo) {
            result += jamo.cho + jamo.jung + jamo.jong;
        } else {
            result += char;
        }
    }
    return result;
}

/**
 * 두 한글 문자의 자소 유사도 계산 (0.0 ~ 1.0)
 */
export function jamoSimilarity(char1: string, char2: string): number {
    const jamo1 = decomposeHangul(char1);
    const jamo2 = decomposeHangul(char2);

    if (!jamo1 || !jamo2) return 0;

    let score = 0;
    if (jamo1.cho === jamo2.cho) score += 0.4;
    if (jamo1.jung === jamo2.jung) score += 0.4;
    if (jamo1.jong === jamo2.jong) score += 0.2;

    return score;
}

/**
 * 한글 유사 자소 매핑 (OCR 오류 패턴)
 */
export const SIMILAR_JAMO: Record<string, string[]> = {
    // 초성 유사
    'ㄱ': ['ㅋ', 'ㄲ'],
    'ㄷ': ['ㅌ', 'ㄸ'],
    'ㅂ': ['ㅍ', 'ㅃ'],
    'ㅅ': ['ㅆ'],
    'ㅈ': ['ㅉ', 'ㅊ'],
    // 중성 유사
    'ㅏ': ['ㅑ', 'ㅓ'],
    'ㅓ': ['ㅏ', 'ㅕ'],
    'ㅗ': ['ㅛ', 'ㅜ'],
    'ㅜ': ['ㅠ', 'ㅗ'],
    'ㅡ': ['ㅢ', 'ㅣ'],
    'ㅣ': ['ㅡ', 'ㅢ'],
};
