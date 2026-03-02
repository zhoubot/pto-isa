/*
 * PTO-BC v0 — Sail formal definition (decode + validation)
 *
 * Scope (v0):
 * - Parse PTO-BC container and sections
 * - Decode key tables (strings/types/attrs/constpool/module)
 * - Decode function bodies enough to validate structure/value-id discipline
 * - Defer full opcode->schema table to a generated artifact (see `opcode_schema_builtin`)
 *
 * This model is intended to be independent of MLIR.
 */

$ifndef PTO_BC_SAIL
$define PTO_BC_SAIL

/*=========================
  Basic types
=========================*/

type byte = bits(8)

type u8  = bits(8)
type u16 = bits(16)
type u32 = bits(32)

type bytes = list(byte)

/*=========================
  Result / cursor
=========================*/

enum result('a, 'e) = { Ok('a), Err('e) }

type err = string

struct cursor = { bs : bytes }

val mk_cursor : bytes -> cursor
function mk_cursor(bs) = { bs = bs }

val remaining : cursor -> nat
function remaining(c) = length(c.bs)

val read_byte : cursor -> result((byte, cursor), err)
function read_byte(c) =
  match c.bs {
    [] => Err("EOF")
  | b :: rest => Ok((b, { bs = rest }))
  }

val read_bytes : cursor -> nat -> result((bytes, cursor), err)
function read_bytes(c, n) =
  if n == 0 then Ok(([], c)) else
  match read_byte(c) {
    Err(e) => Err(e)
  | Ok((b, c1)) =>
      match read_bytes(c1, n - 1) {
        Err(e2) => Err(e2)
      | Ok((bs2, c2)) => Ok((b :: bs2, c2))
      }
  }

/* fixed-width little-endian */
val read_u8 : cursor -> result((u8, cursor), err)
function read_u8(c) =
  match read_byte(c) {
    Err(e) => Err(e)
  | Ok((b, c1)) => Ok((b, c1))
  }

val read_u16_le : cursor -> result((u16, cursor), err)
function read_u16_le(c) =
  match read_byte(c) {
    Err(e) => Err(e)
  | Ok((b0, c1)) =>
      match read_byte(c1) {
        Err(e2) => Err(e2)
      | Ok((b1, c2)) => Ok((b1 @ b0, c2))
      }
  }

val read_u32_le : cursor -> result((u32, cursor), err)
function read_u32_le(c) =
  match read_byte(c) {
    Err(e) => Err(e)
  | Ok((b0, c1)) =>
      match read_byte(c1) {
        Err(e2) => Err(e2)
      | Ok((b1, c2)) =>
          match read_byte(c2) {
            Err(e3) => Err(e3)
          | Ok((b2, c3)) =>
              match read_byte(c3) {
                Err(e4) => Err(e4)
              | Ok((b3, c4)) => Ok((b3 @ b2 @ b1 @ b0, c4))
              }
          }
      }
  }

/*=========================
  LEB128
=========================*/

val u8_to_int : u8 -> int
function u8_to_int(x) = unsigned(x)

val read_uleb128 : cursor -> result((int, cursor), err)
function read_uleb128(c) = read_uleb128_acc(c, 0, 0)

val read_uleb128_acc : cursor -> int -> int -> result((int, cursor), err)
function read_uleb128_acc(c, shift, acc) =
  match read_u8(c) {
    Err(e) => Err(e)
  | Ok((b, c1)) =>
      let chunk : int = unsigned(b[6..0]) in
      let acc2 : int = acc + (chunk << shift) in
      if b[7] == bitone then
        read_uleb128_acc(c1, shift + 7, acc2)
      else
        Ok((acc2, c1))
  }

val read_sleb128 : cursor -> result((int, cursor), err)
function read_sleb128(c) = read_sleb128_acc(c, 0, 0)

val read_sleb128_acc : cursor -> int -> int -> result((int, cursor), err)
function read_sleb128_acc(c, shift, acc) =
  match read_u8(c) {
    Err(e) => Err(e)
  | Ok((b, c1)) =>
      let chunk : int = unsigned(b[6..0]) in
      let acc2 : int = acc + (chunk << shift) in
      if b[7] == bitone then
        read_sleb128_acc(c1, shift + 7, acc2)
      else
        /* sign-extend if sign bit (bit6) set */
        if b[6] == bitone then
          Ok((acc2 - (1 << (shift + 7)), c1))
        else
          Ok((acc2, c1))
  }

/*=========================
  Header / sections
=========================*/

struct ptobc_header = {
  magic : bytes,   /* length 6 */
  version : u16,
  flags : u16,
  payload_len : u32
}

val magic_bytes : bytes
function magic_bytes = [0x50, 0x54, 0x4f, 0x42, 0x43, 0x00]

val parse_header : cursor -> result((ptobc_header, cursor), err)
function parse_header(c) =
  match read_bytes(c, 6) {
    Err(e) => Err(e)
  | Ok((mg, c1)) =>
      match read_u16_le(c1) {
        Err(e2) => Err(e2)
      | Ok((ver, c2)) =>
          match read_u16_le(c2) {
            Err(e3) => Err(e3)
          | Ok((flg, c3)) =>
              match read_u32_le(c3) {
                Err(e4) => Err(e4)
              | Ok((plen, c4)) => Ok(({ magic = mg, version = ver, flags = flg, payload_len = plen }, c4))
              }
          }
      }
  }

struct section_raw = {
  id : u8,
  len : u32,
  data : bytes
}

val parse_section_raw : cursor -> result((section_raw, cursor), err)
function parse_section_raw(c) =
  match read_u8(c) {
    Err(e) => Err(e)
  | Ok((sid, c1)) =>
      match read_u32_le(c1) {
        Err(e2) => Err(e2)
      | Ok((slen, c2)) =>
          let n : nat = unsigned(slen) in
          match read_bytes(c2, n) {
            Err(e3) => Err(e3)
          | Ok((sd, c3)) => Ok(({ id = sid, len = slen, data = sd }, c3))
          }
      }
  }

/*=========================
  Tables
=========================*/

/* STRINGS */

type string_table = list(bytes)

val parse_strings : bytes -> result(string_table, err)
function parse_strings(bs) =
  let c0 = mk_cursor(bs) in
  match read_uleb128(c0) {
    Err(e) => Err(e)
  | Ok((cnt, c1)) =>
      parse_strings_n(c1, cnt)
  }

val parse_strings_n : cursor -> int -> result(string_table, err)
function parse_strings_n(c, n) =
  if n <= 0 then Ok([]) else
  match read_uleb128(c) {
    Err(e) => Err(e)
  | Ok((blen, c1)) =>
      if blen < 0 then Err("negative string length") else
      match read_bytes(c1, (int_to_nat(blen))) {
        Err(e2) => Err(e2)
      | Ok((bstr, c2)) =>
          match parse_strings_n(c2, n - 1) {
            Err(e3) => Err(e3)
          | Ok(rest) => Ok(bstr :: rest)
          }
      }
  }

/* helper: int->nat (partial) */
val int_to_nat : int -> nat
function int_to_nat(x) = if x <= 0 then 0 else (x : nat)

/* TYPES */

enum type_tag = {
  T_Opaque, T_Int, T_Index, T_Float,
  T_PtoPtr, T_PtoTensorView, T_PtoPartitionTensorView, T_PtoTileBuf,
  T_Func
}

/* Structured payloads */
struct dim = { kind : u8, value : int }

struct type_payload = {
  tag : u8,
  has_asm : bool,
  asm_sid : int,
  /* tag-specific fields */
  int_bitwidth : int,
  float_dtype_id : int,
  ptr_elem_type_id : int,
  view_rank : int,
  view_dims : list(dim),
  view_elem_type_id : int,
  pview_rank : int,
  pview_dims_u16 : list(u16),
  pview_elem_type_id : int,
  tile_loc : u8,
  tile_elem_type_id : int,
  tile_rows : u16,
  tile_cols : u16,
  tile_vrow_mode : u8,
  tile_vrow : u16,
  tile_vcol_mode : u8,
  tile_vcol : u16,
  tile_blayout : u8,
  tile_slayout : u8,
  tile_fractal : u16,
  tile_pad : u16,
  func_num_args : int,
  func_arg_types : list(int),
  func_num_results : int,
  func_res_types : list(int)
}

/* In v0 we always keep the MLIR asm backup; structured fields may be zero if unknown. */

type type_table = list(type_payload)

val parse_types : bytes -> result(type_table, err)
function parse_types(bs) =
  let c0 = mk_cursor(bs) in
  match read_uleb128(c0) {
    Err(e) => Err(e)
  | Ok((cnt, c1)) => parse_types_n(c1, cnt)
  }

val parse_types_n : cursor -> int -> result(type_table, err)
function parse_types_n(c, n) =
  if n <= 0 then Ok([]) else
  match parse_type_entry(c) {
    Err(e) => Err(e)
  | Ok((te, c1)) =>
      match parse_types_n(c1, n - 1) {
        Err(e2) => Err(e2)
      | Ok(rest) => Ok(te :: rest)
      }
  }

val parse_type_entry : cursor -> result((type_payload, cursor), err)
function parse_type_entry(c) =
  match read_u8(c) {
    Err(e) => Err(e)
  | Ok((tag, c1)) =>
      match read_u8(c1) {
        Err(e2) => Err(e2)
      | Ok((flags, c2)) =>
          let has_asm : bool = (flags[0] == bitone) in
          if has_asm then
            match read_uleb128(c2) {
              Err(e3) => Err(e3)
            | Ok((sid, c3)) => parse_type_payload(tag, true, sid, c3)
            }
          else
            parse_type_payload(tag, false, -1, c2)
      }
  }

val mk_empty_type : u8 -> bool -> int -> type_payload
function mk_empty_type(tag, has_asm, sid) = {
  tag = tag, has_asm = has_asm, asm_sid = sid,
  int_bitwidth = 0,
  float_dtype_id = 0,
  ptr_elem_type_id = 0,
  view_rank = 0, view_dims = [], view_elem_type_id = 0,
  pview_rank = 0, pview_dims_u16 = [], pview_elem_type_id = 0,
  tile_loc = 0, tile_elem_type_id = 0, tile_rows = 0x0000, tile_cols = 0x0000,
  tile_vrow_mode = 0, tile_vrow = 0x0000, tile_vcol_mode = 0, tile_vcol = 0x0000,
  tile_blayout = 0, tile_slayout = 0, tile_fractal = 0x0000, tile_pad = 0x0000,
  func_num_args = 0, func_arg_types = [], func_num_results = 0, func_res_types = []
}

val parse_type_payload : u8 -> bool -> int -> cursor -> result((type_payload, cursor), err)
function parse_type_payload(tag, has_asm, sid, c) =
  let base = mk_empty_type(tag, has_asm, sid) in
  /* switch on tag */
  if tag == 0x00 then Ok((base, c)) else
  if tag == 0x01 then
    match read_u8(c) {
      Err(e) => Err(e)
    | Ok((bw, c1)) => Ok(({ base with int_bitwidth = unsigned(bw) }, c1))
    }
  else if tag == 0x02 then Ok((base, c))
  else if tag == 0x03 then
    match read_uleb128(c) {
      Err(e) => Err(e)
    | Ok((dt, c1)) => Ok(({ base with float_dtype_id = dt }, c1))
    }
  else if tag == 0x10 then
    match read_uleb128(c) {
      Err(e) => Err(e)
    | Ok((et, c1)) => Ok(({ base with ptr_elem_type_id = et }, c1))
    }
  else if tag == 0x11 then
    match read_uleb128(c) {
      Err(e) => Err(e)
    | Ok((rk, c1)) =>
        match parse_dims(c1, rk) {
          Err(e2) => Err(e2)
        | Ok((ds, c2)) =>
            match read_uleb128(c2) {
              Err(e3) => Err(e3)
            | Ok((et, c3)) => Ok(({ base with view_rank = rk, view_dims = ds, view_elem_type_id = et }, c3))
            }
        }
    }
  else if tag == 0x12 then
    match read_uleb128(c) {
      Err(e) => Err(e)
    | Ok((rk, c1)) =>
        match parse_u16_list(c1, rk) {
          Err(e2) => Err(e2)
        | Ok((ds, c2)) =>
            match read_uleb128(c2) {
              Err(e3) => Err(e3)
            | Ok((et, c3)) => Ok(({ base with pview_rank = rk, pview_dims_u16 = ds, pview_elem_type_id = et }, c3))
            }
        }
    }
  else if tag == 0x13 then
    match read_u8(c) {
      Err(e) => Err(e)
    | Ok((loc, c1)) =>
        match read_uleb128(c1) {
          Err(e2) => Err(e2)
        | Ok((et, c2)) =>
            match read_u16_le(c2) {
              Err(e3) => Err(e3)
            | Ok((rows, c3)) =>
                match read_u16_le(c3) {
                  Err(e4) => Err(e4)
                | Ok((cols, c4)) =>
                    match read_u8(c4) {
                      Err(e5) => Err(e5)
                    | Ok((vrow_mode, c5)) =>
                        match parse_opt_u16(c5, vrow_mode) {
                          Err(e6) => Err(e6)
                        | Ok((vrow, c6)) =>
                            match read_u8(c6) {
                              Err(e7) => Err(e7)
                            | Ok((vcol_mode, c7)) =>
                                match parse_opt_u16(c7, vcol_mode) {
                                  Err(e8) => Err(e8)
                                | Ok((vcol, c8)) =>
                                    match read_u8(c8) {
                                      Err(e9) => Err(e9)
                                    | Ok((bl, c9)) =>
                                        match read_u8(c9) {
                                          Err(e10) => Err(e10)
                                        | Ok((sl, c10)) =>
                                            match read_u16_le(c10) {
                                              Err(e11) => Err(e11)
                                            | Ok((fr, c11)) =>
                                                match read_u16_le(c11) {
                                                  Err(e12) => Err(e12)
                                                | Ok((pad, c12)) =>
                                                    Ok(({ base with tile_loc = loc, tile_elem_type_id = et,
                                                         tile_rows = rows, tile_cols = cols,
                                                         tile_vrow_mode = vrow_mode, tile_vrow = vrow,
                                                         tile_vcol_mode = vcol_mode, tile_vcol = vcol,
                                                         tile_blayout = bl, tile_slayout = sl,
                                                         tile_fractal = fr, tile_pad = pad }, c12))
                                                }
                                            }
                                        }
                                    }
                                }
                        }
                    }
                }
            }
        }
    }
  else if tag == 0x20 then
    match read_uleb128(c) {
      Err(e) => Err(e)
    | Ok((na, c1)) =>
        match parse_uleb_list(c1, na) {
          Err(e2) => Err(e2)
        | Ok((args, c2)) =>
            match read_uleb128(c2) {
              Err(e3) => Err(e3)
            | Ok((nr, c3)) =>
                match parse_uleb_list(c3, nr) {
                  Err(e4) => Err(e4)
                | Ok((ress, c4)) => Ok(({ base with func_num_args = na, func_arg_types = args, func_num_results = nr, func_res_types = ress }, c4))
                }
            }
        }
    }
  else
    /* unknown structured tag: accept only if asm present */
    if has_asm then Ok((base, c)) else Err("type tag without asm")

val parse_dims : cursor -> int -> result((list(dim), cursor), err)
function parse_dims(c, n) =
  if n <= 0 then Ok(([], c)) else
  match read_u8(c) {
    Err(e) => Err(e)
  | Ok((k, c1)) =>
      if k == 0x00 then
        match parse_dims(c1, n - 1) {
          Err(e2) => Err(e2)
        | Ok((rest, c2)) => Ok(({ kind = k, value = 0 } :: rest, c2))
        }
      else
        match read_uleb128(c1) {
          Err(e3) => Err(e3)
        | Ok((v, c2)) =>
            match parse_dims(c2, n - 1) {
              Err(e4) => Err(e4)
            | Ok((rest, c3)) => Ok(({ kind = k, value = v } :: rest, c3))
            }
        }
  }

val parse_u16_list : cursor -> int -> result((list(u16), cursor), err)
function parse_u16_list(c, n) =
  if n <= 0 then Ok(([], c)) else
  match read_u16_le(c) {
    Err(e) => Err(e)
  | Ok((d, c1)) =>
      match parse_u16_list(c1, n - 1) {
        Err(e2) => Err(e2)
      | Ok((rest, c2)) => Ok((d :: rest, c2))
      }
  }

val parse_uleb_list : cursor -> int -> result((list(int), cursor), err)
function parse_uleb_list(c, n) =
  if n <= 0 then Ok(([], c)) else
  match read_uleb128(c) {
    Err(e) => Err(e)
  | Ok((v, c1)) =>
      match parse_uleb_list(c1, n - 1) {
        Err(e2) => Err(e2)
      | Ok((rest, c2)) => Ok((v :: rest, c2))
      }
  }

val parse_opt_u16 : cursor -> u8 -> result((u16, cursor), err)
function parse_opt_u16(c, mode) =
  if mode == 0x00 then read_u16_le(c) else Ok((0x0000, c))

/* ATTRS */

type attr_entry = { tag : u8, asm_sid : int }
type attr_table = list(attr_entry)

val parse_attrs : bytes -> result(attr_table, err)
function parse_attrs(bs) =
  let c0 = mk_cursor(bs) in
  match read_uleb128(c0) {
    Err(e) => Err(e)
  | Ok((cnt, c1)) => parse_attrs_n(c1, cnt)
  }

val parse_attrs_n : cursor -> int -> result(attr_table, err)
function parse_attrs_n(c, n) =
  if n <= 0 then Ok([]) else
  match read_u8(c) {
    Err(e) => Err(e)
  | Ok((tag, c1)) =>
      match read_u8(c1) {
        Err(e2) => Err(e2)
      | Ok((flags, c2)) =>
          if flags[0] != bitone then Err("attr missing asm") else
          match read_uleb128(c2) {
            Err(e3) => Err(e3)
          | Ok((sid, c3)) =>
              match parse_attrs_n(c3, n - 1) {
                Err(e4) => Err(e4)
              | Ok(rest) => Ok(({ tag = tag, asm_sid = sid } :: rest))
              }
          }
      }
  }

/* CONSTPOOL */

enum const_tag = { C_Int, C_FloatBits, C_IndexVec, C_Unknown }

struct const_entry = {
  tag : u8,
  int_type_id : int,
  int_value : int,
  float_dtype_id : int,
  float_bytes : bytes,
  idx_vec : list(int)
}

type const_pool = list(const_entry)

val mk_empty_const : u8 -> const_entry
function mk_empty_const(tag) = {
  tag = tag,
  int_type_id = 0, int_value = 0,
  float_dtype_id = 0, float_bytes = [],
  idx_vec = []
}

val parse_constpool : bytes -> result(const_pool, err)
function parse_constpool(bs) =
  let c0 = mk_cursor(bs) in
  match read_uleb128(c0) {
    Err(e) => Err(e)
  | Ok((cnt, c1)) => parse_const_n(c1, cnt)
  }

val parse_const_n : cursor -> int -> result(const_pool, err)
function parse_const_n(c, n) =
  if n <= 0 then Ok([]) else
  match parse_const_entry(c) {
    Err(e) => Err(e)
  | Ok((ce, c1)) =>
      match parse_const_n(c1, n - 1) {
        Err(e2) => Err(e2)
      | Ok(rest) => Ok(ce :: rest)
      }
  }

val parse_const_entry : cursor -> result((const_entry, cursor), err)
function parse_const_entry(c) =
  match read_u8(c) {
    Err(e) => Err(e)
  | Ok((tag, c1)) =>
      let base = mk_empty_const(tag) in
      if tag == 0x01 then
        match read_uleb128(c1) {
          Err(e2) => Err(e2)
        | Ok((tid, c2)) =>
            match read_sleb128(c2) {
              Err(e3) => Err(e3)
            | Ok((v, c3)) => Ok(({ base with int_type_id = tid, int_value = v }, c3))
            }
        }
      else if tag == 0x02 then
        match read_uleb128(c1) {
          Err(e2) => Err(e2)
        | Ok((dt, c2)) =>
            match read_uleb128(c2) {
              Err(e3) => Err(e3)
            | Ok((blen, c3)) =>
                match read_bytes(c3, int_to_nat(blen)) {
                  Err(e4) => Err(e4)
                | Ok((fb, c4)) => Ok(({ base with float_dtype_id = dt, float_bytes = fb }, c4))
                }
            }
        }
      else if tag == 0x03 then
        match read_uleb128(c1) {
          Err(e2) => Err(e2)
        | Ok((ln, c2)) =>
            match parse_sleb_list(c2, ln) {
              Err(e3) => Err(e3)
            | Ok((xs, c3)) => Ok(({ base with idx_vec = xs }, c3))
            }
        }
      else
        Err("unknown const tag")
  }

val parse_sleb_list : cursor -> int -> result((list(int), cursor), err)
function parse_sleb_list(c, n) =
  if n <= 0 then Ok(([], c)) else
  match read_sleb128(c) {
    Err(e) => Err(e)
  | Ok((v, c1)) =>
      match parse_sleb_list(c1, n - 1) {
        Err(e2) => Err(e2)
      | Ok((rest, c2)) => Ok((v :: rest, c2))
      }
  }

/*=========================
  Module / bodies
=========================*/

struct global_entry = {
  name_sid : int,
  flags : u8,
  kind : u8,
  payload : int
}

struct func_decl = {
  name_sid : int,
  func_type_id : int,
  flags : u8,
  func_attr_id : int
}

struct op_generic = {
  attr_id : int,
  op_name_sid : int,
  result_type_ids : list(int),
  operand_value_ids : list(int),
  regions : list(region)
}

struct op_known = {
  opcode : u16,
  attr_id : int,
  variant_u8 : int,
  immediates : list(int),
  operand_value_ids : list(int),
  explicit_result_type_ids : list(int),
  regions : list(region)
}

enum op = { OpKnown(op_known), OpGeneric(op_generic) }

struct block = {
  arg_type_ids : list(int),
  ops : list(op)
}

struct region = { blocks : list(block) }

struct function_body = { body : region, op_count : int, value_count : int }

struct func_stat = { is_import : bool, op_count : int, value_count : int }

struct module_ast = {
  profile_id : u8,
  index_width : u8,
  module_attr_id : int,
  globals : list(global_entry),
  funcs : list(func_decl),
  func_stats : list(func_stat),
  bodies : list(function_body)
}

/* Parsing state for value_id/op_id validation */
struct fstate = { next_value : int, next_op : int }

val opcode_generic : u16
function opcode_generic = 0xffff

/* Builtin opcode schema table (generated). */
struct opcode_schema = {
  has_variant_u8 : bool,
  result_type_mode : u8,      /* 0=infer, 1=explicit */

  operand_mode : u8,          /* 0=fixed, 1=by_variant, 2=varcount(total-prefix), 3=segmented(view), 4=optional_mask2 */
  num_operands : int,         /* used when operand_mode==0 */

  num_results : int,
  num_regions : int,

  imm_kind : u8               /* 0=none, 1=cmpi_pred(u8), 2=event3(u8,u8,u8), 5=const_id(uLEB128), 6/7=view-segments, 8=alloc_tile optmask */
}

/* Generated builtin opcode schema table (v0). */
$include "generated/pto_bc_opcodes_v0.sail"

val opcode_schema_builtin : u16 -> option(opcode_schema)
function opcode_schema_builtin(op) = opcode_schema_builtin_gen(op)

val opcode_operands_by_variant : u16 -> int -> option(int)
function opcode_operands_by_variant(op, variant) = opcode_operands_by_variant_gen(op, variant)

struct module_stats = { max_value_id : int, max_op_id : int }

val merge_stats : module_stats -> module_stats -> module_stats
function merge_stats(a, b) = {
  max_value_id = if a.max_value_id > b.max_value_id then a.max_value_id else b.max_value_id,
  max_op_id = if a.max_op_id > b.max_op_id then a.max_op_id else b.max_op_id
}

val empty_stats : module_stats
function empty_stats = { max_value_id = 0, max_op_id = 0 }

val parse_module : bytes -> result((module_ast, module_stats), err)
function parse_module(bs) =
  let c0 = mk_cursor(bs) in
  match read_u8(c0) {
    Err(e) => Err(e)
  | Ok((profile, c1)) =>
      match read_u8(c1) {
        Err(e2) => Err(e2)
      | Ok((iw, c2)) =>
          match read_uleb128(c2) {
            Err(e3) => Err(e3)
          | Ok((mattr, c3)) =>
              match read_uleb128(c3) {
                Err(e4) => Err(e4)
              | Ok((gcnt, c4)) =>
                  match parse_globals(c4, gcnt) {
                    Err(e5) => Err(e5)
                  | Ok((gls, c5)) =>
                      match read_uleb128(c5) {
                        Err(e6) => Err(e6)
                      | Ok((fcnt, c6)) =>
                          match parse_func_decls(c6, fcnt) {
                            Err(e7) => Err(e7)
                          | Ok((fds, c7)) =>
                              match parse_bodies(c7, fds) {
                                Err(e8) => Err(e8)
                              | Ok((stats, bds, c8)) =>
                                  /* require full consumption */
                                  if remaining(c8) == 0 then
                                    Ok({ profile_id = profile, index_width = iw, module_attr_id = mattr,
                                         globals = gls, funcs = fds, func_stats = stats, bodies = bds })
                                  else
                                    Err("trailing bytes in MODULE")
                              }
                          }
                      }
                  }
              }
          }
      }
  }

val parse_globals : cursor -> int -> result((list(global_entry), cursor), err)
function parse_globals(c, n) =
  if n <= 0 then Ok(([], c)) else
  match read_uleb128(c) {
    Err(e) => Err(e)
  | Ok((nsid, c1)) =>
      match read_u8(c1) {
        Err(e2) => Err(e2)
      | Ok((fl, c2)) =>
          match read_u8(c2) {
            Err(e3) => Err(e3)
          | Ok((kd, c3)) =>
              match read_uleb128(c3) {
                Err(e4) => Err(e4)
              | Ok((pl, c4)) =>
                  match parse_globals(c4, n - 1) {
                    Err(e5) => Err(e5)
                  | Ok((rest, c5)) => Ok(({ name_sid = nsid, flags = fl, kind = kd, payload = pl } :: rest, c5))
                  }
              }
          }
      }
  }

val parse_func_decls : cursor -> int -> result((list(func_decl), cursor), err)
function parse_func_decls(c, n) =
  if n <= 0 then Ok(([], c)) else
  match read_uleb128(c) {
    Err(e) => Err(e)
  | Ok((nsid, c1)) =>
      match read_uleb128(c1) {
        Err(e2) => Err(e2)
      | Ok((ftid, c2)) =>
          match read_u8(c2) {
            Err(e3) => Err(e3)
          | Ok((fl, c3)) =>
              match read_uleb128(c3) {
                Err(e4) => Err(e4)
              | Ok((faid, c4)) =>
                  match parse_func_decls(c4, n - 1) {
                    Err(e5) => Err(e5)
                  | Ok((rest, c5)) => Ok(({ name_sid = nsid, func_type_id = ftid, flags = fl, func_attr_id = faid } :: rest, c5))
                  }
              }
          }
      }
  }

val is_import_func : func_decl -> bool
function is_import_func(fd) = (fd.flags[0] == bitone)

val parse_bodies : cursor -> list(func_decl) -> result((list(func_stat), list(function_body), cursor), err)
function parse_bodies(c, fds) =
  match fds {
    [] => Ok(([], [], c))
  | fd :: rest =>
      if is_import_func(fd) then
        match parse_bodies(c, rest) {
          Err(e) => Err(e)
        | Ok((srest, brest, c1)) => Ok(({ is_import = true, op_count = 0, value_count = 0 } :: srest, brest, c1))
        }
      else
        match parse_region(c, { next_value = 0, next_op = 0 }) {
          Err(e2) => Err(e2)
        | Ok((reg, st_end, c1)) =>
            match parse_bodies(c1, rest) {
              Err(e3) => Err(e3)
            | Ok((srest, brest, c2)) =>
                Ok(({ is_import = false, op_count = st_end.next_op, value_count = st_end.next_value } :: srest,
                    ({ body = reg, op_count = st_end.next_op, value_count = st_end.next_value } :: brest),
                    c2))
            }
        }
  }

val parse_region : cursor -> fstate -> result((region, fstate, cursor), err)
function parse_region(c, st) =
  match read_uleb128(c) {
    Err(e) => Err(e)
  | Ok((bcnt, c1)) =>
      match parse_blocks(c1, st, bcnt) {
        Err(e2) => Err(e2)
      | Ok((blks, st2, c2)) => Ok(({ blocks = blks }, st2, c2))
      }
  }

val parse_blocks : cursor -> fstate -> int -> result((list(block), fstate, cursor), err)
function parse_blocks(c, st, n) =
  if n <= 0 then Ok(([], st, c)) else
  match parse_block(c, st) {
    Err(e) => Err(e)
  | Ok((b, st1, c1)) =>
      match parse_blocks(c1, st1, n - 1) {
        Err(e2) => Err(e2)
      | Ok((rest, st2, c2)) => Ok((b :: rest, st2, c2))
      }
  }

val parse_block : cursor -> fstate -> result((block, fstate, cursor), err)
function parse_block(c, st) =
  match read_uleb128(c) {
    Err(e) => Err(e)
  | Ok((nargs, c1)) =>
      match parse_uleb_list(c1, nargs) {
        Err(e2) => Err(e2)
      | Ok((arg_tys, c2)) =>
          /* allocate value_ids for block args */
          let st_args = { st with next_value = st.next_value + nargs } in
          match read_uleb128(c2) {
            Err(e3) => Err(e3)
          | Ok((opcnt, c3)) =>
              match parse_ops(c3, st_args, opcnt) {
                Err(e4) => Err(e4)
              | Ok((ops, st4, c4)) => Ok(({ arg_type_ids = arg_tys, ops = ops }, st4, c4))
              }
          }
      }
  }

val parse_ops : cursor -> fstate -> int -> result((list(op), fstate, cursor), err)
function parse_ops(c, st, n) =
  if n <= 0 then Ok(([], st, c)) else
  match parse_op(c, st) {
    Err(e) => Err(e)
  | Ok((op1, st1, c1)) =>
      match parse_ops(c1, st1, n - 1) {
        Err(e2) => Err(e2)
      | Ok((rest, st2, c2)) => Ok((op1 :: rest, st2, c2))
      }
  }

val parse_op : cursor -> fstate -> result((op, fstate, cursor), err)
function parse_op(c, st) =
  match read_u16_le(c) {
    Err(e) => Err(e)
  | Ok((opc, c1)) =>
      match read_uleb128(c1) {
        Err(e2) => Err(e2)
      | Ok((attr, c2)) =>
          if opc == opcode_generic then
            parse_generic_op(c2, st, attr)
          else
            parse_known_op(c2, st, opc, attr)
      }
  }

val parse_generic_op : cursor -> fstate -> int -> result((op, fstate, cursor), err)
function parse_generic_op(c, st, attr) =
  match read_uleb128(c) {
    Err(e) => Err(e)
  | Ok((name_sid, c1)) =>
      match read_uleb128(c1) {
        Err(e2) => Err(e2)
      | Ok((nr, c2)) =>
          match parse_uleb_list(c2, nr) {
            Err(e3) => Err(e3)
          | Ok((rtys, c3)) =>
              match read_uleb128(c3) {
                Err(e4) => Err(e4)
              | Ok((no, c4)) =>
                  match parse_uleb_list(c4, no) {
                    Err(e5) => Err(e5)
                  | Ok((ops, c5)) =>
                      /* validate operands refer to previous values */
                      if all_lt(ops, st.next_value) then
                        let st_res = { st with next_value = st.next_value + nr, next_op = st.next_op + 1 } in
                        match read_uleb128(c5) {
                          Err(e6) => Err(e6)
                        | Ok((nreg, c6)) =>
                            match parse_regions(c6, st_res, nreg) {
                              Err(e7) => Err(e7)
                            | Ok((regs, st7, c7)) =>
                                Ok((OpGeneric({ attr_id = attr, op_name_sid = name_sid,
                                                result_type_ids = rtys,
                                                operand_value_ids = ops,
                                                regions = regs }), st7, c7))
                            }
                        }
                      else
                        Err("operand value_id out of range")
                  }
              }
          }
      }
  }

val parse_known_operands : cursor -> u16 -> opcode_schema -> int -> list(int) -> result((list(int), cursor), err)
function parse_known_operands(c, opc, sch, variant, imms) =
  if sch.operand_mode == 0x00 then
    parse_uleb_list(c, sch.num_operands)

  else if sch.operand_mode == 0x01 then
    match opcode_operands_by_variant(opc, variant) {
      None() => Err("missing by-variant operand count")
    | Some(n) => parse_uleb_list(c, n)
    }

  else if sch.operand_mode == 0x02 then
    match read_uleb128(c) {
      Err(e) => Err(e)
    | Ok((n, c1)) => parse_uleb_list(c1, n)
    }

  else if sch.operand_mode == 0x03 then
    /* segmented view forms: list_mode + 2 segment lengths */
    match imms {
      lm :: n1 :: n2 :: _ =>
        let base : int = sch.num_operands in
        if lm == 0 then
          parse_uleb_list(c, base + n1 + n2)
        else
          /* list_mode==1 encodes const_vector_ids for the two lists */
          parse_uleb_list(c, base + 2)
    | _ => Err("segmented operands missing immediates")
    }

  else if sch.operand_mode == 0x04 then
    /* optional mask for (valid_row?, valid_col?) */
    match imms {
      mask :: _ =>
        let row : int = mask mod 2 in
        let col : int = (mask / 2) mod 2 in
        parse_uleb_list(c, row + col)
    | _ => Err("optional-mask operands missing immediates")
    }

  else
    Err("unknown operand_mode")

val parse_known_op : cursor -> fstate -> u16 -> int -> result((op, fstate, cursor), err)
function parse_known_op(c, st, opc, attr) =
  match opcode_schema_builtin(opc) {
    None() => Err("missing opcode schema")
  | Some(sch) =>
      /* immediates */
      match parse_known_imms(c, sch) {
        Err(e) => Err(e)
      | Ok((variant, imms, c1)) =>
          /* operands */
          match parse_known_operands(c1, opc, sch, variant, imms) {
            Err(e2) => Err(e2)
          | Ok((ops, c2)) =>
              if all_lt(ops, st.next_value) then
                /* explicit result types if required */
                match parse_explicit_result_types(c2, sch) {
                  Err(e3) => Err(e3)
                | Ok((rtys, c3)) =>
                    let st_res = { st with next_value = st.next_value + sch.num_results, next_op = st.next_op + 1 } in
                    match parse_regions(c3, st_res, sch.num_regions) {
                      Err(e4) => Err(e4)
                    | Ok((regs, st4, c4)) =>
                        Ok((OpKnown({ opcode = opc, attr_id = attr, variant_u8 = variant,
                                      immediates = imms, operand_value_ids = ops,
                                      explicit_result_type_ids = rtys, regions = regs }), st4, c4))
                    }
                }
              else
                Err("operand value_id out of range")
          }
      }
  }

val parse_known_imms : cursor -> opcode_schema -> result((int, list(int), cursor), err)
function parse_known_imms(c, sch) =
  /* variant */
  if sch.has_variant_u8 then
    match read_u8(c) {
      Err(e) => Err(e)
    | Ok((v, c1)) => parse_known_imms2(c1, sch, unsigned(v), [])
    }
  else
    parse_known_imms2(c, sch, 0, [])

val parse_known_imms2 : cursor -> opcode_schema -> int -> list(int) -> result((int, list(int), cursor), err)
function parse_known_imms2(c, sch, variant, acc) =
  /* imm_kind:
     0=none
     1=cmpi_pred(u8)
     2=event3(u8,u8,u8)
     5=const_id(uLEB128)
     6=make_tensor_view: list_mode(u8), nshape(uLEB128), nstrides(uLEB128)
     7=partition_view: list_mode(u8), noffsets(uLEB128), nsizes(uLEB128)
     8=alloc_tile: opt_mask(u8)
  */
  if sch.imm_kind == 0x00 then Ok((variant, reverse(acc), c))
  else if sch.imm_kind == 0x01 then
    match read_u8(c) {
      Err(e) => Err(e)
    | Ok((p, c1)) => Ok((variant, reverse(unsigned(p) :: acc), c1))
    }
  else if sch.imm_kind == 0x02 then
    match read_u8(c) {
      Err(e) => Err(e)
    | Ok((a, c1)) =>
        match read_u8(c1) {
          Err(e2) => Err(e2)
        | Ok((b, c2)) =>
            match read_u8(c2) {
              Err(e3) => Err(e3)
            | Ok((d, c3)) => Ok((variant, reverse(unsigned(d) :: unsigned(b) :: unsigned(a) :: acc), c3))
            }
        }
    }
  else if sch.imm_kind == 0x05 then
    match read_uleb128(c) {
      Err(e) => Err(e)
    | Ok((cid, c1)) => Ok((variant, reverse(cid :: acc), c1))
    }
  else if sch.imm_kind == 0x06 then
    match read_u8(c) {
      Err(e) => Err(e)
    | Ok((lm, c1)) =>
        match read_uleb128(c1) {
          Err(e2) => Err(e2)
        | Ok((n1, c2)) =>
            match read_uleb128(c2) {
              Err(e3) => Err(e3)
            | Ok((n2, c3)) => Ok((variant, reverse(unsigned(lm) :: n1 :: n2 :: acc), c3))
            }
        }
    }
  else if sch.imm_kind == 0x07 then
    match read_u8(c) {
      Err(e) => Err(e)
    | Ok((lm, c1)) =>
        match read_uleb128(c1) {
          Err(e2) => Err(e2)
        | Ok((n1, c2)) =>
            match read_uleb128(c2) {
              Err(e3) => Err(e3)
            | Ok((n2, c3)) => Ok((variant, reverse(unsigned(lm) :: n1 :: n2 :: acc), c3))
            }
        }
    }
  else if sch.imm_kind == 0x08 then
    match read_u8(c) {
      Err(e) => Err(e)
    | Ok((m, c1)) => Ok((variant, reverse(unsigned(m) :: acc), c1))
    }
  else
    Err("unknown imm_kind")

val parse_explicit_result_types : cursor -> opcode_schema -> result((list(int), cursor), err)
function parse_explicit_result_types(c, sch) =
  if sch.result_type_mode == 0x01 then
    parse_uleb_list(c, sch.num_results)
  else
    Ok(([], c))

val parse_regions : cursor -> fstate -> int -> result((list(region), fstate, cursor), err)
function parse_regions(c, st, n) =
  if n <= 0 then Ok(([], st, c)) else
  match parse_region(c, st) {
    Err(e) => Err(e)
  | Ok((r, st1, c1)) =>
      match parse_regions(c1, st1, n - 1) {
        Err(e2) => Err(e2)
      | Ok((rest, st2, c2)) => Ok((r :: rest, st2, c2))
      }
  }

val all_lt : list(int) -> int -> bool
function all_lt(xs, bound) =
  match xs {
    [] => true
  | x :: rest => (x >= 0) & (x < bound) & all_lt(rest, bound)
  }

/*=========================
  Top-level PTOBC file
=========================*/

struct dbg_file = { path_sid : int, hash : bytes }
struct dbg_value_name = { func_id : int, value_id : int, name_sid : int }
struct dbg_location = { func_id : int, op_id : int, file_id : int, sl : int, sc : int, el : int, ec : int }
struct dbg_snippet = { func_id : int, op_id : int, snippet_sid : int }

struct debuginfo_ast = {
  files : list(dbg_file),
  value_names : list(dbg_value_name),
  locations : list(dbg_location),
  snippets : list(dbg_snippet)
}

struct ptobc_ast = {
  header : ptobc_header,
  strings : string_table,
  types : type_table,
  attrs : attr_table,
  consts : const_pool,
  module : module_ast,
  debuginfo : option(debuginfo_ast),
  extra_sections : list(section_raw)
}

val validate_debuginfo : module_ast -> option(debuginfo_ast) -> int -> result(unit, err)
function validate_debuginfo(mo, dbg_opt, nstrings) =
  match dbg_opt {
    None() => Ok(())
  | Some(dbg) =>
      let nfiles : int = length(dbg.files) in
      /* validate file paths and ids */
      if all_sid_lt(dbg.files, nstrings) & all_vname_ok(dbg.value_names, mo.func_stats, nstrings) &
         all_loc_ok(dbg.locations, mo.func_stats, nfiles) & all_snip_ok(dbg.snippets, mo.func_stats, nstrings) then
        Ok(())
      else
        Err("debuginfo cross-validation failed")
  }

val all_sid_lt : list(dbg_file) -> int -> bool
function all_sid_lt(fs, nstrings) =
  match fs {
    [] => true
  | f :: rest => (f.path_sid >= 0) & (f.path_sid < nstrings) & all_sid_lt(rest, nstrings)
  }

val stat_ok : list(func_stat) -> int -> bool
function stat_ok(stats, fid) = (fid >= 0) & (fid < length(stats))

val get_stat : list(func_stat) -> int -> func_stat
function get_stat(stats, fid) =
  match stats {
    [] => { is_import = true, op_count = 0, value_count = 0 }
  | s :: rest => if fid == 0 then s else get_stat(rest, fid - 1)
  }

val all_vname_ok : list(dbg_value_name) -> list(func_stat) -> int -> bool
function all_vname_ok(vns, stats, nstrings) =
  match vns {
    [] => true
  | vn :: rest =>
      if stat_ok(stats, vn.func_id) then
        let st = get_stat(stats, vn.func_id) in
        (st.is_import == false) & (vn.value_id >= 0) & (vn.value_id < st.value_count) &
        (vn.name_sid >= 0) & (vn.name_sid < nstrings) & all_vname_ok(rest, stats, nstrings)
      else
        false
  }

val all_loc_ok : list(dbg_location) -> list(func_stat) -> int -> bool
function all_loc_ok(ls, stats, nfiles) =
  match ls {
    [] => true
  | l :: rest =>
      if stat_ok(stats, l.func_id) then
        let st = get_stat(stats, l.func_id) in
        (st.is_import == false) & (l.op_id >= 0) & (l.op_id < st.op_count) &
        (l.file_id >= 0) & (l.file_id < nfiles) & all_loc_ok(rest, stats, nfiles)
      else
        false
  }

val all_snip_ok : list(dbg_snippet) -> list(func_stat) -> int -> bool
function all_snip_ok(ss, stats, nstrings) =
  match ss {
    [] => true
  | s :: rest =>
      if stat_ok(stats, s.func_id) then
        let st = get_stat(stats, s.func_id) in
        (st.is_import == false) & (s.op_id >= 0) & (s.op_id < st.op_count) &
        (s.snippet_sid >= 0) & (s.snippet_sid < nstrings) & all_snip_ok(rest, stats, nstrings)
      else
        false
  }

val validate_debuginfo_ids : option(debuginfo_ast) -> module_stats -> result(unit, err)
function validate_debuginfo_ids(dbg_opt, st) =
  match dbg_opt {
    None() => Ok(())
  | Some(dbg) =>
      /* file_id bounds */
      let fcnt : int = length(dbg.files) in
      if all_loc_file_lt(dbg.locations, fcnt) then
        if all_opid_lt(dbg.locations, st.max_op_id) & all_opid_lt_snip(dbg.snippets, st.max_op_id) then
          if all_vid_lt(dbg.value_names, st.max_value_id) then
            Ok(())
          else Err("DEBUGINFO value_id out of range")
        else Err("DEBUGINFO op_id out of range")
      else Err("DEBUGINFO file_id out of range")
  }

val all_loc_file_lt : list(dbg_location) -> int -> bool
function all_loc_file_lt(locs, bound) =
  match locs {
    [] => true
  | l :: rest => (l.file_id >= 0) & (l.file_id < bound) & all_loc_file_lt(rest, bound)
  }

val all_opid_lt : list(dbg_location) -> int -> bool
function all_opid_lt(locs, bound) =
  match locs {
    [] => true
  | l :: rest => (l.op_id >= 0) & (l.op_id < bound) & all_opid_lt(rest, bound)
  }

val all_opid_lt_snip : list(dbg_snippet) -> int -> bool
function all_opid_lt_snip(snips, bound) =
  match snips {
    [] => true
  | s :: rest => (s.op_id >= 0) & (s.op_id < bound) & all_opid_lt_snip(rest, bound)
  }

val all_vid_lt : list(dbg_value_name) -> int -> bool
function all_vid_lt(vns, bound) =
  match vns {
    [] => true
  | v :: rest => (v.value_id >= 0) & (v.value_id < bound) & all_vid_lt(rest, bound)
  }

val parse_ptobc_v0 : bytes -> result(ptobc_ast, err)
function parse_ptobc_v0(bs) =
  let c0 = mk_cursor(bs) in
  match parse_header(c0) {
    Err(e) => Err(e)
  | Ok((hd, c1)) =>
      /* strict header validation */
      if hd.magic != magic_bytes then Err("bad magic") else
      if hd.version != 0x0000 then Err("unsupported version") else
      if hd.flags != 0x0000 then Err("flags must be 0 in v0") else
      let plen : nat = unsigned(hd.payload_len) in
      if remaining(c1) != plen then Err("payload_len mismatch") else
      /* parse fixed sections in order */
      match parse_section_raw(c1) {
        Err(e2) => Err(e2)
      | Ok((s_strings, c2)) =>
          if s_strings.id != 0x01 then Err("missing STRINGS") else
          match parse_section_raw(c2) {
            Err(e3) => Err(e3)
          | Ok((s_types, c3)) =>
              if s_types.id != 0x02 then Err("missing TYPES") else
              match parse_section_raw(c3) {
                Err(e4) => Err(e4)
              | Ok((s_attrs, c4)) =>
                  if s_attrs.id != 0x03 then Err("missing ATTRS") else
                  match parse_section_raw(c4) {
                    Err(e5) => Err(e5)
                  | Ok((s_const, c5)) =>
                      if s_const.id != 0x04 then Err("missing CONSTPOOL") else
                      /* optional schema_ext */
                      match parse_optional_schema_ext(c5) {
                        Err(e6) => Err(e6)
                      | Ok((c6, _has_schema_ext)) =>
                          match parse_section_raw(c6) {
                            Err(e7) => Err(e7)
                          | Ok((s_mod, c7)) =>
                              if s_mod.id != 0x06 then Err("missing MODULE") else
                              /* optional debuginfo */
                              match parse_optional_debuginfo(c7) {
                                Err(e8) => Err(e8)
                              | Ok((c8, dbg_opt, extra_raw)) =>
                                  /* decode tables */
                                  match parse_strings(s_strings.data) {
                                    Err(e9) => Err(e9)
                                  | Ok(strs) =>
                                      match parse_types(s_types.data) {
                                        Err(e10) => Err(e10)
                                      | Ok(tys) =>
                                          match parse_attrs(s_attrs.data) {
                                            Err(e11) => Err(e11)
                                          | Ok(at) =>
                                              match parse_constpool(s_const.data) {
                                                Err(e12) => Err(e12)
                                              | Ok(cs) =>
                                                  match parse_module(s_mod.data) {
                                                    Err(e13) => Err(e13)
                                                  | Ok(mo) =>
                                                      if remaining(c8) != 0 then Err("trailing bytes after sections") else
                                                      match validate_debuginfo(mo, dbg_opt, length(strs)) {
                                                        Err(e14) => Err(e14)
                                                      | Ok(_) =>
                                                          Ok({ header = hd, strings = strs, types = tys, attrs = at,
                                                               consts = cs, module = mo,
                                                               debuginfo = dbg_opt,
                                                               extra_sections = extra_raw })
                                                      }
                                                  }
                                              }
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
          }
      }
  }

val parse_optional_schema_ext : cursor -> result((cursor, bool), err)
function parse_optional_schema_ext(c) =
  /* If next section is OPCODE_SCHEMA_EXT (0x05), skip it for now. */
  if remaining(c) == 0 then Ok((c, false)) else
  match parse_section_raw(c) {
    Err(e) => Err(e)
  | Ok((s, c1)) =>
      if s.id == 0x05 then Ok((c1, true)) else Ok((c, false))
  }

val parse_debuginfo : bytes -> result(debuginfo_ast, err)
function parse_debuginfo(bs) =
  let c0 = mk_cursor(bs) in
  match read_uleb128(c0) {
    Err(e) => Err(e)
  | Ok((fcnt, c1)) =>
      match parse_dbg_files(c1, fcnt) {
        Err(e2) => Err(e2)
      | Ok((files, c2)) =>
          match read_uleb128(c2) {
            Err(e3) => Err(e3)
          | Ok((vcnt, c3)) =>
              match parse_dbg_value_names(c3, vcnt) {
                Err(e4) => Err(e4)
              | Ok((vns, c4)) =>
                  match read_uleb128(c4) {
                    Err(e5) => Err(e5)
                  | Ok((lcnt, c5)) =>
                      match parse_dbg_locations(c5, lcnt) {
                        Err(e6) => Err(e6)
                      | Ok((locs, c6)) =>
                          match read_uleb128(c6) {
                            Err(e7) => Err(e7)
                          | Ok((scnt, c7)) =>
                              match parse_dbg_snippets(c7, scnt) {
                                Err(e8) => Err(e8)
                              | Ok((snips, c8)) =>
                                  if remaining(c8) == 0 then
                                    Ok({ files = files, value_names = vns, locations = locs, snippets = snips })
                                  else
                                    Err("trailing bytes in DEBUGINFO")
                              }
                          }
                      }
                  }
              }
          }
      }
  }

val parse_dbg_files : cursor -> int -> result((list(dbg_file), cursor), err)
function parse_dbg_files(c, n) =
  if n <= 0 then Ok(([], c)) else
  match read_uleb128(c) {
    Err(e) => Err(e)
  | Ok((psid, c1)) =>
      match read_uleb128(c1) {
        Err(e2) => Err(e2)
      | Ok((hlen, c2)) =>
          match read_bytes(c2, int_to_nat(hlen)) {
            Err(e3) => Err(e3)
          | Ok((hb, c3)) =>
              match parse_dbg_files(c3, n - 1) {
                Err(e4) => Err(e4)
              | Ok((rest, c4)) => Ok(({ path_sid = psid, hash = hb } :: rest, c4))
              }
          }
      }
  }

val parse_dbg_value_names : cursor -> int -> result((list(dbg_value_name), cursor), err)
function parse_dbg_value_names(c, n) =
  if n <= 0 then Ok(([], c)) else
  match read_uleb128(c) {
    Err(e) => Err(e)
  | Ok((fid, c1)) =>
      match read_uleb128(c1) {
        Err(e2) => Err(e2)
      | Ok((vid, c2)) =>
          match read_uleb128(c2) {
            Err(e3) => Err(e3)
          | Ok((nsid, c3)) =>
              match parse_dbg_value_names(c3, n - 1) {
                Err(e4) => Err(e4)
              | Ok((rest, c4)) => Ok(({ func_id = fid, value_id = vid, name_sid = nsid } :: rest, c4))
              }
          }
      }
  }

val parse_dbg_locations : cursor -> int -> result((list(dbg_location), cursor), err)
function parse_dbg_locations(c, n) =
  if n <= 0 then Ok(([], c)) else
  match read_uleb128(c) {
    Err(e) => Err(e)
  | Ok((fn, c1)) =>
      match read_uleb128(c1) {
        Err(e2) => Err(e2)
      | Ok((opid, c2)) =>
          match read_uleb128(c2) {
            Err(e3) => Err(e3)
          | Ok((fid, c3)) =>
              match read_uleb128(c3) {
                Err(e4) => Err(e4)
              | Ok((sl, c4)) =>
                  match read_uleb128(c4) {
                    Err(e5) => Err(e5)
                  | Ok((sc, c5)) =>
                      match read_uleb128(c5) {
                        Err(e6) => Err(e6)
                      | Ok((el, c6)) =>
                          match read_uleb128(c6) {
                            Err(e7) => Err(e7)
                          | Ok((ec, c7)) =>
                              match parse_dbg_locations(c7, n - 1) {
                                Err(e8) => Err(e8)
                              | Ok((rest, c8)) => Ok(({ func_id = fn, op_id = opid, file_id = fid, sl = sl, sc = sc, el = el, ec = ec } :: rest, c8))
                              }
                          }
                      }
                  }
              }
          }
      }
  }

val parse_dbg_snippets : cursor -> int -> result((list(dbg_snippet), cursor), err)
function parse_dbg_snippets(c, n) =
  if n <= 0 then Ok(([], c)) else
  match read_uleb128(c) {
    Err(e) => Err(e)
  | Ok((fn, c1)) =>
      match read_uleb128(c1) {
        Err(e2) => Err(e2)
      | Ok((opid, c2)) =>
          match read_uleb128(c2) {
            Err(e3) => Err(e3)
          | Ok((ssid, c3)) =>
              match parse_dbg_snippets(c3, n - 1) {
                Err(e4) => Err(e4)
              | Ok((rest, c4)) => Ok(({ func_id = fn, op_id = opid, snippet_sid = ssid } :: rest, c4))
              }
          }
      }
  }

val parse_optional_debuginfo : cursor -> result((cursor, option(debuginfo_ast), list(section_raw)), err)
function parse_optional_debuginfo(c) =
  if remaining(c) == 0 then Ok((c, None(), [])) else
  match parse_section_raw(c) {
    Err(e) => Err(e)
  | Ok((s, c1)) =>
      if s.id == 0x07 then
        match parse_debuginfo(s.data) {
          Err(e2) => Err(e2)
        | Ok(dbg) =>
            match parse_optional_extra(c1) {
              Err(e3) => Err(e3)
            | Ok((c2, xs)) => Ok((c2, Some(dbg), xs))
            }
        }
      else
        match parse_optional_extra(c) {
          Err(e3) => Err(e3)
        | Ok((c2, xs)) => Ok((c2, None(), xs))
        }
  }

val parse_optional_extra : cursor -> result((cursor, list(section_raw)), err)
function parse_optional_extra(c) =
  if remaining(c) == 0 then Ok((c, [])) else
  match parse_section_raw(c) {
    Err(e) => Err(e)
  | Ok((s, c1)) =>
      if s.id != 0x7F then Err("unknown section outside EXTRA") else
        /* parse nested raw sections inside EXTRA */
        match parse_extra_payload(s.data) {
          Err(e2) => Err(e2)
        | Ok(xs) => Ok((c1, xs))
        }
  }

val parse_extra_payload : bytes -> result((list(section_raw)), err)
function parse_extra_payload(bs) =
  let c0 = mk_cursor(bs) in
  parse_extra_sections(c0)

val parse_extra_sections : cursor -> result((list(section_raw)), err)
function parse_extra_sections(c) =
  if remaining(c) == 0 then Ok([]) else
  match parse_section_raw(c) {
    Err(e) => Err(e)
  | Ok((s, c1)) =>
      match parse_extra_sections(c1) {
        Err(e2) => Err(e2)
      | Ok(rest) => Ok(s :: rest)
      }
  }

$endif
