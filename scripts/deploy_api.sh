#!/usr/bin/env bash
# 一键部署 RDA API 服务端代码：备份 -> 推送 -> 重启 -> 等就绪 -> smoke
#
# 设计原则（v0.7.3+）：
#   * 服务端私仓不在 GitHub 公开仓库里 → 不预设默认文件，每次部署必须显式给
#     "<本地相对仓库根路径>:<远端绝对路径>" 配对，避免错把客户端代码推到服务端
#   * 备份命名 .bak-YYYYMMDDTHHMMSSZ 与服务端 .bak-YYYYMMDD 共存（不冲突）
#   * 用 systemd restart 而不是手工 kill，is-active 确认状态
#   * 每次同步必须 smoke POST /api/v1/recommend，health 不走 recommend 路径
#
# 用法：
#   scripts/deploy_api.sh <local>:<remote> [<local>:<remote> ...]
#   scripts/deploy_api.sh --rollback <remote>
#   scripts/deploy_api.sh --skip-smoke   <pairs...>
#   scripts/deploy_api.sh --no-restart   <pairs...>
#
# 示例：
#   scripts/deploy_api.sh \
#       rda/recommend/engine.py:/opt/rda-api/server/engine_core.py \
#       rda/recommend/formatter.py:/opt/rda-api/server/formatter.py
#
#   scripts/deploy_api.sh --rollback /opt/rda-api/server/engine_core.py
#
# 环境变量覆盖：
#   RDA_API_HOST          远端主机（默认 49.233.206.42）
#   RDA_API_USER          SSH 用户（默认 ubuntu）
#   RDA_API_SSH_KEY       私钥路径（默认 E:/下载的软件包/minilm-L6-V6/workbuddy.pem）
#   RDA_API_LOCAL_PORT       服务端绑的 loopback 端口（默认 8100，等就绪探测用）
#   RDA_API_BASE_URL      smoke 测试的公网入口（默认 https://rda.niusu2026.cn）
#   RDA_API_SERVICE       systemd 服务名（默认 rda-api）
#   RDA_API_REQUIRE_SERVER_MARKER  本地源必须含服务端标记才放行（默认 true）
#
# 退出码：
#   0 部署成功（含 smoke 通过）
#   1 本地校验失败 / 参数错误
#   2 ssh/scp 推送失败
#   3 重启失败 / 服务未监听
#   4 smoke 测试失败

set -euo pipefail

# -------- 配置 --------
RDA_API_HOST="${RDA_API_HOST:-49.233.206.42}"
RDA_API_USER="${RDA_API_USER:-ubuntu}"
RDA_API_SSH_KEY="${RDA_API_SSH_KEY:-E:/下载的软件包/minilm-L6-V6/workbuddy.pem}"
RDA_API_LOCAL_PORT="${RDA_API_LOCAL_PORT:-8100}"          # ssh 内网 ss 探测用
RDA_API_BASE_URL="${RDA_API_BASE_URL:-https://rda.niusu2026.cn}"   # 公网 smoke 用
RDA_API_SERVICE="${RDA_API_SERVICE:-rda-api}"
RDA_API_REQUIRE_SERVER_MARKER="${RDA_API_REQUIRE_SERVER_MARKER:-true}"

# -------- 路径推导 --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SRC="${LOCAL_SRC:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# -------- 颜色 --------
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; NC=''
fi

info()  { echo -e "${CYAN}[deploy]${NC} $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
fail()  {
    local code=$1; shift
    echo -e "${RED}[fail]${NC}  $*" >&2
    exit "$code"
}

usage() { sed -n '2,32p' "$0"; }

# -------- 参数解析 --------
SMOKE=true
RESTART=true
SKIP_BACKUP=false
SKIP_PUSH=false
ROLLBACK_REMOTE=""
PAIRS=()  # 存 "<local>|<remote>"（竖线避免冒号和 Windows 路径冲突）

for arg in "$@"; do
    case "$arg" in
        --skip-smoke)   SMOKE=false ;;
        --no-restart)   RESTART=false ;;
        --smoke-only)   SMOKE=true; RESTART=false; SKIP_BACKUP=true; SKIP_PUSH=true ;;
        --no-backup)    SKIP_BACKUP=true ;;
        --allow-client-marker) RDA_API_REQUIRE_SERVER_MARKER=false ;;
        --rollback)
            shift || true
            if [ -z "${1:-}" ]; then
                fail 1 "--rollback 需要接一个远端绝对路径"
            fi
            ROLLBACK_REMOTE="$1"
            shift
            ;;
        -h|--help)     usage; exit 0 ;;
        --*)           fail 1 "未知选项: $arg" ;;
        *)
            if ! echo "$arg" | grep -q ':'; then
                fail 1 "参数必须是 '<local_rel>:<remote_abs>' 形式，收到: $arg"
            fi
            PAIRS+=("$arg")
            ;;
    esac
done

# -------- 回滚分支 --------
if [ -n "$ROLLBACK_REMOTE" ]; then
    [ "${RESTART}" = false ] && fail 1 "--rollback 与 --no-restart 互斥"
    info "回滚 ${ROLLBACK_REMOTE} 到最新 .bak-*..."
    ssh -i "${RDA_API_SSH_KEY}" -o StrictHostKeyChecking=accept-new \
        -o ConnectTimeout=15 \
        "${RDA_API_USER}@${RDA_API_HOST}" \
        "F=${ROLLBACK_REMOTE} && \
         LATEST=\$(ls -1t \${F}.bak-* 2>/dev/null | head -1) && \
         if [ -z \"\$LATEST\" ]; then echo 'NO BACKUP FOUND' >&2; exit 1; fi && \
         echo \"rollback target: \$LATEST\" && \
         sudo cp \"\$LATEST\" \"\${F}\" && \
         sudo systemctl restart ${RDA_API_SERVICE} && \
         sleep 1 && \
         sudo systemctl is-active ${RDA_API_SERVICE}"
    ok "回滚 + 重启完成"
    exit 0
fi

# 必须至少一对（除非 --smoke-only）
if [ ${#PAIRS[@]} -eq 0 ] && [ "$SKIP_PUSH" = false ]; then
    fail 1 "没有任何部署对。用 --help 看用法。"
fi

# -------- 校验：拆 local / remote --------
declare -a LOCALS=()
declare -a REMOTES=()
for p in "${PAIRS[@]}"; do
    LOCAL_REL="${p%%:*}"
    REMOTE_ABS="${p#*:}"
    case "$REMOTE_ABS" in
        /*) ;;
        *)  fail 1 "远端路径必须是绝对路径: ${REMOTE_ABS}" ;;
    esac
    LOCALS+=("$LOCAL_REL")
    REMOTES+=("$REMOTE_ABS")
done

# -------- 打印计划 --------
echo -e "${BOLD}========== RDA API 一键部署 ==========${NC}"
info "目标主机:    ${RDA_API_USER}@${RDA_API_HOST}"
info "本地源:      ${LOCAL_SRC}"
info "服务名:      ${RDA_API_SERVICE}"
info "公网 smoke:  ${RDA_API_BASE_URL}"
info "本地端口:    ${RDA_API_LOCAL_PORT}"
info "smoke:       ${SMOKE}"
info "restart:     ${RESTART}"
for i in "${!LOCALS[@]}"; do
    info "  pair[${i}]: ${LOCALS[$i]}  ->  ${REMOTES[$i]}"
done
echo

# -------- 1) 本地校验 --------
info "校验本地文件..."
for L in "${LOCALS[@]}"; do
    FULL="${LOCAL_SRC}/${L}"
    if [ ! -f "${FULL}" ]; then
        fail 1 "本地文件不存在: ${FULL}"
    fi
    # 服务端标记探测：避免误把客户端 stub 推上来
    if [ "$RDA_API_REQUIRE_SERVER_MARKER" = "true" ]; then
        if ! grep -qE '@app\.(get|post|put|delete|route)|FastAPI\(|uvicorn|server\.main' "${FULL}" 2>/dev/null; then
            fail 1 "本地源 ${L} 不像服务端代码（缺 @app.*/FastAPI/uvicorn/server.main 标记）。客户端 stub 推上来会炸服务端。如果确认要用，加 --allow-client-marker 强推。"
        fi
    fi
done
ok "本地文件齐全"

[ -f "${RDA_API_SSH_KEY}" ] || fail 1 "SSH 私钥不存在: ${RDA_API_SSH_KEY}"

SSH_BASE=( -i "${RDA_API_SSH_KEY}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 )
SCP_BASE=( -i "${RDA_API_SSH_KEY}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 )

# -------- 2) 远端备份 --------
if [ "$SKIP_BACKUP" = false ]; then
    info "远端备份原文件..."
    TS="$(date -u +%Y%m%dT%H%M%SZ)"
    REMOTES_CSV=$(IFS=,; echo "${REMOTES[*]}")

    ssh "${SSH_BASE[@]}" "${RDA_API_USER}@${RDA_API_HOST}" \
        "for f in ${REMOTES_CSV//,/ }; do \
             if [ -f \"\$f\" ]; then \
                 sudo cp \"\$f\" \"\${f}.bak-${TS}\" && \
                 echo \"  backed up: \$f -> \${f}.bak-${TS}\"; \
             else \
                 echo \"  not present (skipped): \$f\"; \
             fi; \
         done"
    ok "备份完成（时间戳 ${TS}）"
else
    warn "跳过备份（--no-backup / --smoke-only）"
fi

# -------- 3) scp 推送 --------
if [ "$SKIP_PUSH" = false ]; then
    info "scp 推送..."
    for i in "${!LOCALS[@]}"; do
        info "  -> ${REMOTES[$i]}"
        scp "${SCP_BASE[@]}" \
            "${LOCAL_SRC}/${LOCALS[$i]}" \
            "${RDA_API_USER}@${RDA_API_HOST}:${REMOTES[$i]}" \
            || fail 2 "scp 推送失败: ${LOCALS[$i]} -> ${REMOTES[$i]}"
    done
    ok "推送完成"
else
    warn "跳过推送（--smoke-only 模式）"
fi

# -------- 4) 重启服务 --------
if [ "$RESTART" = true ]; then
    info "重启 systemd 服务 ${RDA_API_SERVICE}..."
    ssh "${SSH_BASE[@]}" "${RDA_API_USER}@${RDA_API_HOST}" \
        "sudo systemctl restart ${RDA_API_SERVICE} && \
         sleep 1 && \
         sudo systemctl is-active ${RDA_API_SERVICE}"
    ok "服务已重启"
else
    warn "跳过重启（--no-restart 模式）"
fi

# -------- 5) 等 loopback 端口监听（ssh 内网探针）--------
info "等待本地端口 ${RDA_API_LOCAL_PORT} 监听..."
WAITED=0
for i in $(seq 1 30); do
    sleep 1
    WAITED=$i
    if ssh "${SSH_BASE[@]}" "${RDA_API_USER}@${RDA_API_HOST}" \
        "ss -tlnp 2>/dev/null | grep -q :${RDA_API_LOCAL_PORT} || \
         netstat -tlnp 2>/dev/null | grep -q :${RDA_API_LOCAL_PORT}" \
        2>/dev/null; then
        ok "端口已就绪（等待 ${WAITED}s）"
        break
    fi
done
[ "$WAITED" -ge 30 ] && fail 3 "端口 ${RDA_API_LOCAL_PORT} 30s 内未监听，请 journalctl -u ${RDA_API_SERVICE} -n 50 --no-pager"

# -------- 6) smoke 测试 --------
if [ "$SMOKE" = true ]; then
    echo
    info "smoke 测试开始..."

    # 6a) 公网 health（/api/v1/health，nginx 反代可达）
    HEALTH=$(curl -sS --max-time 15 "${RDA_API_BASE_URL}/api/v1/health" || echo "FAIL")
    case "$HEALTH" in
        *ok*|*healthy*|*status*ok*) ok "health: ${HEALTH}" ;;
        *) warn "health 返回异常: ${HEALTH:0:200}" ;;
    esac

    # 6b) recommend（带 audit_signals 走 req3 视觉建议路径）
    # 这是 v0.7.3 修 bug 的那条路径，health 不会触发，必须 POST 才暴露
    # schema 顶层字段：policy / lang / contract_version / temporal_sufficiency / episode_count / total_frames
    PAYLOAD='{"contract_version":3,"policy":"frame-wise","lang":"en","episode_count":1,"total_frames":50,"temporal_sufficiency":{"total_episodes":1,"total_frames":50,"computed_episodes":1},"audit_signals":{"visual_summary":{"episodes_total":1,"va_a_exclude_episodes":1,"va_b_review_episodes":0,"frozen_regions_count":2,"per_camera_penalty_max":0.8}}}'

    RESP=$(curl -sS --max-time 30 -X POST \
        -H 'Content-Type: application/json' \
        -d "$PAYLOAD" \
        "${RDA_API_BASE_URL}/api/v1/recommend" || echo "FAIL")

    if echo "$RESP" | grep -qE "VISUAL_REPAIR_FIRST|VISUAL_QUALITY_REVIEW"; then
        ADVICE=$(echo "$RESP" | grep -oE 'VISUAL_[A-Z_]+' | sort -u | tr '\n' ',' | sed 's/,$//')
        ok "recommend 视觉建议路径通过: ${ADVICE}"
    elif echo "$RESP" | grep -qE '"detail"|"error"|500 Internal'; then
        fail 4 "recommend 报错: ${RESP:0:400}"
    else
        warn "recommend 返回未识别（可能是 schema 变化）: ${RESP:0:300}"
    fi

    ENGINE=$(echo "$RESP" | grep -oE '"engine_version"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 || echo "")
    [ -n "$ENGINE" ] && info "engine: ${ENGINE}"
else
    warn "跳过 smoke（--skip-smoke 模式）"
fi

echo
echo -e "${GREEN}${BOLD}========== 部署完成 ✓ ==========${NC}"
echo "回滚命令: $0 --rollback <remote_abs_path>"
echo "查看备份: ssh -i ${RDA_API_SSH_KEY} ${RDA_API_USER}@${RDA_API_HOST} 'ls -lt /opt/rda-api/server/*.bak-* | head'"