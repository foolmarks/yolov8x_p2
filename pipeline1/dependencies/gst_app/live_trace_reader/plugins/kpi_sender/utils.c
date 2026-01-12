#include <glib.h>
#include <babeltrace2/babeltrace.h>
#include <stdio.h>
#include <malloc.h>
#include <ctype.h>
#include <simaai/simaailog.h>

#include "utils.h"

/*
 * Produce a build-time error if the condition `cond` is non-zero.
 * Evaluates as a size_t expression.
 */
#define _BT_BUILD_ASSERT(cond)					\
	sizeof(struct { int f:(2 * !!(cond) - 1); })

/*
 * Cast value `v` to an unsigned integer of the same size as `v`.
 */
#define _bt_cast_value_to_unsigned(v)					\
	(sizeof(v) == sizeof(uint8_t) ? (uint8_t) (v) :			\
	sizeof(v) == sizeof(uint16_t) ? (uint16_t) (v) :		\
	sizeof(v) == sizeof(uint32_t) ? (uint32_t) (v) :		\
	sizeof(v) == sizeof(uint64_t) ? (uint64_t) (v) :		\
	_BT_BUILD_ASSERT(sizeof(v) <= sizeof(uint64_t)))

/*
 * Left shift a value `v` of `shift` bits.
 *
 * The type of `v` can be signed or unsigned integer.
 * The value of `shift` must be less than the size of `v` (in bits),
 * otherwise the behavior is undefined.
 * Evaluates to the result of the shift operation.
 *
 * According to the C99 standard, left shift of a left hand-side signed
 * type is undefined if it has a negative value or if the result cannot
 * be represented in the result type. This bitfield header discards the
 * bits that are left-shifted beyond the result type representation,
 * which is the behavior of an unsigned type left shift operation.
 * Therefore, always perform left shift on an unsigned type.
 *
 * This macro should not be used if `shift` can be greater or equal than
 * the bitwidth of `v`. See `_bt_safe_lshift`.
 */
#define _bt_lshift(v, shift)						\
	((__typeof__(v)) (_bt_cast_value_to_unsigned(v) << (shift)))

/*
 * Left shift a signed or unsigned integer with `shift` value being an
 * arbitrary number of bits. `v` is modified by this macro. The shift
 * is transformed into a sequence of `_nr_partial_shifts` consecutive
 * shift operations, each of a number of bits smaller than the bitwidth
 * of `v`, ending with a shift of the number of left over bits.
 */
#define _bt_safe_lshift(v, shift)					\
do {									\
	unsigned long _nr_partial_shifts = (shift) / (sizeof(v) * CHAR_BIT - 1); \
	unsigned long _leftover_bits = (shift) % (sizeof(v) * CHAR_BIT - 1); \
									\
	for (; _nr_partial_shifts; _nr_partial_shifts--)		\
		(v) = _bt_lshift(v, sizeof(v) * CHAR_BIT - 1);		\
	(v) = _bt_lshift(v, _leftover_bits);				\
} while (0)

int parse_field(const bt_field *field, GString *result);

static int parse_string(const bt_field *field, GString *result)
{
	const char *str = bt_field_string_get_value(field);
	if (!str) {
		return -1;
	}

	for (int i = 0; i < strlen(str); i++) {
		/* Escape sequences not recognized by iscntrl(). */
		switch (str[i]) {
		case '\\':
			g_string_append(result, "\\\\");
			continue;
		case '\'':
			g_string_append(result, "\\\'");
			continue;
		case '\"':
			g_string_append(result, "\\\"");
			continue;
		case '\?':
			g_string_append(result, "\\\?");
			continue;
		}

		/* Standard characters. */
		if (!iscntrl((unsigned char) str[i])) {
			g_string_append_c(result, str[i]);
			continue;
		}

		switch (str[i]) {
		case '\0':
			g_string_append(result, "\\0");
			break;
		case '\a':
			g_string_append(result, "\\a");
			break;
		case '\b':
			g_string_append(result, "\\b");
			break;
		case '\e':
			g_string_append(result, "\\e");
			break;
		case '\f':
			g_string_append(result, "\\f");
			break;
		case '\n':
			g_string_append(result, "\\n");
			break;
		case '\r':
			g_string_append(result, "\\r");
			break;
		case '\t':
			g_string_append(result, "\\t");
			break;
		case '\v':
			g_string_append(result, "\\v");
			break;
		default:
			/* Unhandled control-sequence, print as hex. */
			g_string_append_printf(result, "\\x%02x", str[i]);
			break;
		}
	}

	return 0;
}

int get_payload_field_by_name(const bt_field *payload, const char *name, const bt_field **result)
{
	if (payload == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot get payload field by name: payload is NULL");
		return -1;
	}
	if (result == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot get payload field by name: result string pointer is NULL");
		return -1;
	}
	if (name == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot get payload field by name: name is NULL");
		return -1;
	}

	*result = bt_field_structure_borrow_member_field_by_name_const(payload, name);
	return 0;
}


int parse_uint64_field(const bt_field *field, uint64_t *result)
{
	if (result == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot parse uint64_t field: result string pointer is NULL");
		return -1;
	}

	bt_field_class_type ft_type = bt_field_get_class_type(field);
	if (bt_field_class_type_is(ft_type, BT_FIELD_CLASS_TYPE_UNSIGNED_INTEGER)) {
		*result = bt_field_integer_unsigned_get_value(field);
	} else {
        simaailog(SIMAAILOG_ERR, "Cannot parse uint64_t field: class type does not match to BT_FIELD_CLASS_TYPE_UNSIGNED_INTEGER");
		return -1;
	}

	return 0;
}

static int parse_integer(const bt_field *field, GString *result)
{
	int ret = 0;
	union {
		uint64_t u;
		int64_t s;
	} v;

	const bt_field_class *int_fc = bt_field_borrow_class_const(field);
	if (int_fc == NULL) {
		return -1;
	}

	bt_field_class_type ft_type = bt_field_get_class_type(field);
	if (bt_field_class_type_is(ft_type, BT_FIELD_CLASS_TYPE_UNSIGNED_INTEGER)) {
		v.u = bt_field_integer_unsigned_get_value(field);
	} else {
		v.s = bt_field_integer_signed_get_value(field);
	}

	bt_field_class_integer_preferred_display_base base = bt_field_class_integer_get_preferred_display_base(int_fc);
	switch (base) {
	case BT_FIELD_CLASS_INTEGER_PREFERRED_DISPLAY_BASE_BINARY:
	{
		g_string_append(result, "0b");

		int len = bt_field_class_integer_get_field_value_range(int_fc);
		_bt_safe_lshift(v.u, 64 - len);
		for (int bitnr = 0; bitnr < len; bitnr++) {
			g_string_append_c(result, (v.u & (1ULL << 63)) ? '1' : '0');
			_bt_safe_lshift(v.u, 1);
		}
		break;
	}
	case BT_FIELD_CLASS_INTEGER_PREFERRED_DISPLAY_BASE_OCTAL:
	{
		if (bt_field_class_type_is(ft_type, BT_FIELD_CLASS_TYPE_SIGNED_INTEGER)) {
			int len = bt_field_class_integer_get_field_value_range(int_fc);
			if (len < 64) {
				if (len == 0) {
					return -1;
				}

				/* Round length to the nearest 3-bit */
				size_t rounded_len = (((len - 1) / 3) + 1) * 3;
				v.u &= ((uint64_t) 1 << rounded_len) - 1;
			}
		}

		g_string_append_printf(result, "%lo", v.u);
		break;
	}
	case BT_FIELD_CLASS_INTEGER_PREFERRED_DISPLAY_BASE_DECIMAL:
		if (bt_field_class_type_is(ft_type, BT_FIELD_CLASS_TYPE_UNSIGNED_INTEGER)) {
			g_string_append_printf(result, "%lu", v.u);
		} else {
			g_string_append_printf(result, "%ld", v.s);
		}
		break;
	case BT_FIELD_CLASS_INTEGER_PREFERRED_DISPLAY_BASE_HEXADECIMAL:
	{
		int len = bt_field_class_integer_get_field_value_range(int_fc);
		if (len < 64) {
			/* Round length to the nearest nibble */
			uint8_t rounded_len = ((len + 3) & ~0x3);
			v.u &= ((uint64_t) 1 << rounded_len) - 1;
		}

		g_string_append_printf(result, "0x%lX", v.u);
		break;
	}
	default:
		ret = -1;
	}

	return ret;
}

int parse_field(const bt_field *field, GString *result)
{
	if (result == NULL) {
        simaailog(SIMAAILOG_ERR, "Cannot parse field: result string pointer is NULL");
		return -1;
	}

	bt_field_class_type class_id;

	class_id = bt_field_get_class_type(field);
	if (!class_id) {
        simaailog(SIMAAILOG_ERR, "Cannot parse field: Unknown field class type");
		return -1;
	}

	if (class_id == BT_FIELD_CLASS_TYPE_BOOL) {
		bt_bool v = bt_field_bool_get_value(field);
		const char *text = v ? "true" : "false";

		g_string_append(result, text);
		return 0;
	} else if (bt_field_class_type_is(class_id, BT_FIELD_CLASS_TYPE_INTEGER)) {
		return parse_integer(field, result);
	} else if (bt_field_class_type_is(class_id, BT_FIELD_CLASS_TYPE_STRING)) {
		return parse_string(field, result);
	} else {
		g_string_append_printf(result, "Error: Unhandled field type: %ld", class_id);
        simaailog(SIMAAILOG_ERR, "Cannot parse field: Unhandled field type: %ld", class_id);
		return -1;
	}

	return -1;
}

uint32_t hash_string_to_uint32(const char *str)
{
	if (str == NULL) {
		return 0;
	}

    uint32_t hash = 2166136261u;
    while (*str) {
        hash ^= (uint8_t)(*str);
        hash *= 16777619u;
        str++;
    }
    return hash;
}

bt_bool is_remote_core_kpi(const plugin_kpi_t * const kpi) {
	bt_bool result = BT_FALSE;
	if (strcmp(kpi->plugin_type, PLUGIN_TYPE_CVU)  == 0 ||
		strcmp(kpi->plugin_type, PLUGIN_TYPE_EV74) == 0 ||
		strcmp(kpi->plugin_type, PLUGIN_TYPE_MLA)  == 0 ||
		strcmp(kpi->plugin_type, PLUGIN_TYPE_M4)   == 0)
	{
		result = BT_TRUE;
	}

	return result;
}
